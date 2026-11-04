# リリース（R8）の規則。縮小と最適化だけを行い、名前は変えない。
# 名前を変えると、JNI の関数名・ネイティブから呼ぶメソッド名・エラー文のクラス名（javaClass.simpleName）が変わるため。
-dontobfuscate

# ネイティブ（jni_bridge.cpp）が GetMethodID で名前を引いて呼ぶメソッド。Kotlin からは呼ばれないので、残す指定が要る
-keep interface com.dragonspeedxl.app.engine.NativeEngine$Listener { *; }
-keep interface com.dragonspeedxl.app.engine.NativeEngine$PreviewListener { *; }
