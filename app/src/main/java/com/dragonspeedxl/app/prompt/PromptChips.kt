package com.dragonspeedxl.app.prompt

import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToLong

/**
 * プロンプトのチップ。姉妹プロジェクト SSXL の webui（prompt_chips.js）と同じ仕様の移植:
 *   label    … 表示名（"hatsune miku"）
 *   booruKey … Danbooru のタグ名（"hatsune_miku"。raw のときは空）
 *   promptText … プロンプトに書く本文（括弧はエスケープ済み）
 *   weight   … 強調の重み（1.0 なら無印）。`(本文:1.20)` で書き出す
 *   raw      … 解析できなかった書式（[…]、<lora:…>、BREAK など）。そのまま保持する
 */
data class PromptChip(
    val id: String,
    val label: String,
    val booruKey: String,
    val promptText: String,
    val weight: Double,
    val raw: Boolean,
    val original: String,
)

data class ParseResult(val chips: List<PromptChip>, val added: Int, val weightCount: Int, val rawCount: Int)

object PromptChips {
    private var seq = 0L
    private fun nextId() = "chip_${++seq}"

    fun weightIsOne(weight: Double) = abs(weight - 1.0) < 1e-9

    fun formatWeight(weight: Double): String = String.format(Locale.US, "%.2f", (weight * 100).roundToLong() / 100.0)

    /** 空欄は 1.0、数値でなければ null。小数 2 桁に丸める。 */
    fun normalizeWeight(value: String?): Double? {
        val t = value?.trim().orEmpty()
        if (t.isEmpty()) return 1.0
        val n = t.toDoubleOrNull() ?: return null
        if (n.isNaN() || n.isInfinite()) return null
        return (n * 100).roundToLong() / 100.0
    }

    fun escapeSd(label: String): String = label.replace("(", "\\(").replace(")", "\\)")
    fun unescapeSd(text: String): String = text.replace("\\(", "(").replace("\\)", ")")
    fun toBooruKey(label: String): String = label.trim().replace(' ', '_')
    fun toLabel(booruKey: String): String = booruKey.replace('_', ' ')

    private val customSyntax = Regex("""\[|<lora:|<lyco:|BREAK\b""", RegexOption.IGNORE_CASE)
    private val hasEscapedParen = Regex("""\\[()]""")
    private val numberText = Regex("""^-?\d+(\.\d+)?$""")

    private fun looksCustomSyntax(text: String) = customSyntax.containsMatchIn(text)

    /** 括弧の外にあるカンマで分ける（`(a, b:1.2)` のような括弧内のカンマでは分けない。`\(` `\)` は無視）。 */
    fun splitTopLevel(text: String): List<String> {
        val parts = ArrayList<String>()
        val buf = StringBuilder()
        var depth = 0
        var escape = false
        for (ch in text) {
            if (escape) { buf.append(ch); escape = false; continue }
            when {
                ch == '\\' -> { buf.append(ch); escape = true }
                ch == '(' -> { depth++; buf.append(ch) }
                ch == ')' -> { depth = maxOf(0, depth - 1); buf.append(ch) }
                ch == ',' && depth == 0 -> { buf.toString().trim().takeIf { it.isNotEmpty() }?.let(parts::add); buf.setLength(0) }
                else -> buf.append(ch)
            }
        }
        buf.toString().trim().takeIf { it.isNotEmpty() }?.let(parts::add)
        return parts
    }

    /** `(本文:1.2)` なら (本文, 1.2)。そうでなければ null。 */
    private fun matchOuterWeight(segment: String): Pair<String, Double>? {
        val s = segment.trim()
        if (!(s.startsWith("(") && s.endsWith(")"))) return null
        val inner = s.substring(1, s.length - 1)
        var depth = 0
        var escape = false
        var lastColon = -1
        for (i in inner.indices) {
            val ch = inner[i]
            if (escape) { escape = false; continue }
            when {
                ch == '\\' -> escape = true
                ch == '(' -> depth++
                ch == ')' -> depth = maxOf(0, depth - 1)
                ch == ':' && depth == 0 -> lastColon = i
            }
        }
        if (lastColon < 0) return null
        val num = inner.substring(lastColon + 1).trim()
        if (!numberText.matches(num)) return null
        val weight = num.toDoubleOrNull() ?: return null
        return inner.substring(0, lastColon).trim() to weight
    }

    private fun createChip(label: String = "", booruKey: String = "", promptText: String = "", weight: Double = 1.0, raw: Boolean = false, original: String = ""): PromptChip =
        PromptChip(nextId(), label, booruKey, promptText, weight, raw, original.ifEmpty { promptText })

    fun chipFromDanbooru(name: String): PromptChip {
        val key = name.trim()
        val label = toLabel(key)
        return createChip(label = label, booruKey = key, promptText = escapeSd(label))
    }

    private fun parseSegment(segment: String): PromptChip? {
        val original = segment.trim()
        if (original.isEmpty()) return null
        val weighted = matchOuterWeight(original)
        val body = weighted?.first ?: original
        val weight = weighted?.second ?: 1.0
        val wrapped = original.startsWith("(") && original.endsWith(")")
        val hasParenButNoWeight = weighted == null && wrapped && body.contains(":")
        val isCustomOrPlainParen = looksCustomSyntax(body) || (weighted == null && wrapped && !body.contains(":"))
        if (body.isEmpty() || hasParenButNoWeight || isCustomOrPlainParen) {
            val lower = body.lowercase()
            if (lower == "negpip" || lower == "negative pi-prompt") return createChip(label = body, promptText = body, original = original)
            if (hasParenButNoWeight) return createChip(label = body, promptText = body, original = original)
            return createChip(label = original, promptText = original, raw = true, original = original)   // 解析できない書式はそのまま保持
        }
        val label: String
        val booruKey: String
        val promptText: String
        when {
            hasEscapedParen.containsMatchIn(body) -> { label = unescapeSd(body); promptText = escapeSd(label); booruKey = toBooruKey(label) }
            body.contains('_') && body.none { it.isWhitespace() } -> { booruKey = body; label = toLabel(body); promptText = escapeSd(label) }
            else -> { label = unescapeSd(body); promptText = escapeSd(label); booruKey = toBooruKey(label) }
        }
        return createChip(label, booruKey, promptText, weight, false, original)
    }

    fun parsePrompt(text: String): ParseResult {
        val chips = ArrayList<PromptChip>()
        var weightCount = 0
        var rawCount = 0
        for (segment in splitTopLevel(text)) {
            val chip = parseSegment(segment) ?: continue
            chips.add(chip)
            if (chip.raw) rawCount++ else if (!weightIsOne(chip.weight)) weightCount++
        }
        return ParseResult(chips, chips.size, weightCount, rawCount)
    }

    fun serializeChip(chip: PromptChip): String {
        if (chip.raw) return chip.original.ifEmpty { chip.promptText.ifEmpty { chip.label } }
        val body = chip.promptText.ifEmpty { escapeSd(chip.label) }
        return if (weightIsOne(chip.weight)) body else "($body:${formatWeight(chip.weight)})"
    }

    fun serializeChips(chips: List<PromptChip>): String = chips.map(::serializeChip).filter { it.isNotEmpty() }.joinToString(", ")

    fun updateChipLabel(chip: PromptChip, nextLabel: String): PromptChip {
        val label = nextLabel.trim()
        if (label.isEmpty()) return chip
        return if (chip.raw) chip.copy(label = label, promptText = label, original = label)
        else chip.copy(label = label, booruKey = toBooruKey(label), promptText = escapeSd(label))
    }

    fun pasteSummary(added: Int, weightCount: Int, rawCount: Int) = "${added}個のチップを追加しました。weight指定: ${weightCount}件、未解析: ${rawCount}件"
}
