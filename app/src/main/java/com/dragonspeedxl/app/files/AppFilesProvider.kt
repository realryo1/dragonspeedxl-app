package com.dragonspeedxl.app.files

import android.content.res.AssetFileDescriptor
import android.database.Cursor
import android.database.MatrixCursor
import android.graphics.Point
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract.Document
import android.provider.DocumentsContract.Root
import android.provider.DocumentsProvider
import android.webkit.MimeTypeMap
import com.dragonspeedxl.app.R
import com.dragonspeedxl.app.security.AppLock
import java.io.File
import java.io.FileNotFoundException

/**
 * アプリのフォルダを Android 標準の「ファイル」アプリ（サイドバー）から見られるようにする DocumentsProvider。
 * Android 11 以降は Android/data が標準では見えないので、モデルの出し入れ（数 GB の .bin のコピー）や履歴画像の取り出しをここから行う。
 * ルートは 1 つ（dragonspeedXL）で、その直下に 2 つのフォルダを並べる:
 *   models  = getExternalFilesDir(null)/models   （モデルを置く）
 *   history = filesDir/history                   （生成した画像）
 * ドキュメント ID は "root"（仮想のルート）または "<マウント名>:<マウントからの相対パス>"。
 */
class AppFilesProvider : DocumentsProvider() {
    private class Mount(val name: String, val dir: File)

    private val mounts: List<Mount> by lazy {
        val c = context!!
        listOf(
            Mount("models", File(c.getExternalFilesDir(null) ?: c.filesDir, "models")),
            Mount("history", File(c.filesDir, "history")),
        ).onEach { it.dir.mkdirs() }
    }

    /**
     * 見せてよいフォルダ。アプリのロックが有効なあいだは、履歴（生成した画像）を出さない。
     * ロックを解かないと見られない、という約束を、システムの「ファイル」アプリからも守るため。
     */
    private fun visibleMounts(): List<Mount> = if (AppLock(context!!).enabled) mounts.filter { it.name != HISTORY } else mounts

    override fun onCreate(): Boolean = true

    private fun mountOf(name: String) = visibleMounts().firstOrNull { it.name == name } ?: throw FileNotFoundException("不明なフォルダ: $name")

    /** ドキュメント ID → File。仮想ルート（"root"）は null。マウントの外へ出る相対パスは拒否する。 */
    private fun fileFor(docId: String): File? {
        if (docId == ROOT_DOC) return null
        val i = docId.indexOf(':')
        if (i < 0) throw FileNotFoundException("不正な ID: $docId")
        val m = mountOf(docId.substring(0, i))
        val f = File(m.dir, docId.substring(i + 1)).canonicalFile
        val base = m.dir.canonicalFile
        if (f != base && !f.path.startsWith(base.path + File.separator)) throw SecurityException("フォルダの外です: $docId")
        return f
    }

    private fun idFor(file: File): String {
        val p = file.canonicalPath
        for (m in visibleMounts()) {
            val base = m.dir.canonicalPath
            if (p == base) return "${m.name}:"
            if (p.startsWith(base + File.separator)) return "${m.name}:" + p.substring(base.length + 1)
        }
        throw FileNotFoundException(p)
    }

    private fun mimeOf(f: File): String {
        if (f.isDirectory) return Document.MIME_TYPE_DIR
        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(f.extension.lowercase()) ?: "application/octet-stream"
    }

    private fun addRoot(c: MatrixCursor) {
        c.newRow()
            .add(Document.COLUMN_DOCUMENT_ID, ROOT_DOC)
            .add(Document.COLUMN_DISPLAY_NAME, "dragonspeedXL")
            .add(Document.COLUMN_SIZE, 0L)
            .add(Document.COLUMN_MIME_TYPE, Document.MIME_TYPE_DIR)
            .add(Document.COLUMN_LAST_MODIFIED, null)
            .add(Document.COLUMN_FLAGS, 0)     // 直下に直接作成はできない（models / history の中に作る）
    }

    private fun addRow(c: MatrixCursor, docId: String, file: File) {
        val isMountRoot = docId.endsWith(":")
        val mime = mimeOf(file)
        var flags = 0
        if (file.isDirectory) {
            flags = flags or Document.FLAG_DIR_SUPPORTS_CREATE
            if (!isMountRoot) flags = flags or Document.FLAG_SUPPORTS_DELETE or Document.FLAG_SUPPORTS_RENAME
        } else {
            flags = flags or Document.FLAG_SUPPORTS_WRITE or Document.FLAG_SUPPORTS_DELETE or Document.FLAG_SUPPORTS_RENAME
            if (mime.startsWith("image/")) flags = flags or Document.FLAG_SUPPORTS_THUMBNAIL
        }
        c.newRow()
            .add(Document.COLUMN_DOCUMENT_ID, docId)
            .add(Document.COLUMN_DISPLAY_NAME, if (isMountRoot) docId.removeSuffix(":") else file.name)
            .add(Document.COLUMN_SIZE, if (file.isDirectory) 0L else file.length())
            .add(Document.COLUMN_MIME_TYPE, mime)
            .add(Document.COLUMN_LAST_MODIFIED, file.lastModified())
            .add(Document.COLUMN_FLAGS, flags)
    }

    override fun queryRoots(projection: Array<out String>?): Cursor {
        val c = MatrixCursor(projection ?: DEFAULT_ROOT_PROJECTION)
        c.newRow()
            .add(Root.COLUMN_ROOT_ID, "app")
            // SUPPORTS_CREATE が無いと、コピー・移動先の選択画面（サイドバー）にルートが出ない
            .add(Root.COLUMN_FLAGS, Root.FLAG_SUPPORTS_CREATE or Root.FLAG_SUPPORTS_IS_CHILD or Root.FLAG_LOCAL_ONLY)
            .add(Root.COLUMN_ICON, R.drawable.ic_launcher)
            .add(Root.COLUMN_TITLE, "dragonspeedXL")
            .add(Root.COLUMN_SUMMARY, if (visibleMounts().size > 1) "models / history" else "models")
            .add(Root.COLUMN_DOCUMENT_ID, ROOT_DOC)
            .add(Root.COLUMN_AVAILABLE_BYTES, mounts.first().dir.usableSpace)
        return c
    }

    override fun queryDocument(documentId: String, projection: Array<out String>?): Cursor {
        val c = MatrixCursor(projection ?: DEFAULT_DOC_PROJECTION)
        val f = fileFor(documentId)
        if (f == null) addRoot(c) else addRow(c, documentId, f)
        return c
    }

    override fun queryChildDocuments(parentDocumentId: String, projection: Array<out String>?, sortOrder: String?): Cursor {
        val c = MatrixCursor(projection ?: DEFAULT_DOC_PROJECTION)
        val dir = fileFor(parentDocumentId)
        if (dir == null) {
            visibleMounts().forEach { addRow(c, "${it.name}:", it.dir) }
        } else {
            dir.listFiles()?.sortedWith(compareBy({ !it.isDirectory }, { it.name.lowercase() }))?.forEach { addRow(c, idFor(it), it) }
        }
        return c
    }

    override fun isChildDocument(parentDocumentId: String, documentId: String): Boolean = try {
        val p = fileFor(parentDocumentId)
        val d = fileFor(documentId)
        when {
            p == null -> true                                   // 仮想ルートの下はすべて子
            d == null -> false
            else -> d.canonicalPath == p.canonicalPath || d.canonicalPath.startsWith(p.canonicalPath + File.separator)
        }
    } catch (_: Exception) {
        false
    }

    private fun realFile(documentId: String): File = fileFor(documentId) ?: throw FileNotFoundException("フォルダは開けません")

    override fun openDocument(documentId: String, mode: String, signal: CancellationSignal?): ParcelFileDescriptor =
        ParcelFileDescriptor.open(realFile(documentId), ParcelFileDescriptor.parseMode(mode))

    override fun openDocumentThumbnail(documentId: String, sizeHint: Point?, signal: CancellationSignal?): AssetFileDescriptor {
        val pfd = ParcelFileDescriptor.open(realFile(documentId), ParcelFileDescriptor.MODE_READ_ONLY)
        return AssetFileDescriptor(pfd, 0, AssetFileDescriptor.UNKNOWN_LENGTH)
    }

    override fun createDocument(parentDocumentId: String, mimeType: String, displayName: String): String {
        val parent = realFile(parentDocumentId)
        val name = File(displayName).name.ifEmpty { "新規" }
        var f = File(parent, name)
        var n = 1
        while (f.exists()) {   // 同名があれば " (1)" を付ける
            val dot = name.lastIndexOf('.')
            f = if (dot > 0 && mimeType != Document.MIME_TYPE_DIR) File(parent, name.substring(0, dot) + " ($n)" + name.substring(dot)) else File(parent, "$name ($n)")
            n++
        }
        if (mimeType == Document.MIME_TYPE_DIR) {
            if (!f.mkdir()) throw FileNotFoundException("作成できません: ${f.name}")
        } else if (!f.createNewFile()) {
            throw FileNotFoundException("作成できません: ${f.name}")
        }
        return idFor(f)
    }

    override fun deleteDocument(documentId: String) {
        if (documentId.endsWith(":") || documentId == ROOT_DOC) throw SecurityException("このフォルダは削除できません")
        val f = realFile(documentId)
        if (!f.deleteRecursively()) throw FileNotFoundException("削除できません: ${f.name}")
    }

    override fun renameDocument(documentId: String, displayName: String): String {
        if (documentId.endsWith(":") || documentId == ROOT_DOC) throw SecurityException("このフォルダは名前を変えられません")
        val f = realFile(documentId)
        val to = File(f.parentFile, File(displayName).name)
        if (to.exists() || !f.renameTo(to)) throw FileNotFoundException("名前を変更できません")
        return idFor(to)
    }

    private companion object {
        const val ROOT_DOC = "root"
        const val HISTORY = "history"
        val DEFAULT_ROOT_PROJECTION = arrayOf(
            Root.COLUMN_ROOT_ID, Root.COLUMN_FLAGS, Root.COLUMN_ICON, Root.COLUMN_TITLE, Root.COLUMN_SUMMARY,
            Root.COLUMN_DOCUMENT_ID, Root.COLUMN_AVAILABLE_BYTES,
        )
        val DEFAULT_DOC_PROJECTION = arrayOf(
            Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME, Document.COLUMN_SIZE, Document.COLUMN_MIME_TYPE,
            Document.COLUMN_LAST_MODIFIED, Document.COLUMN_FLAGS,
        )
    }
}
