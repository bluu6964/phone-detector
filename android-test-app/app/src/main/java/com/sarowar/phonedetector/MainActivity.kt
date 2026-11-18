package com.sarowar.phonedetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.view.Gravity
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Minimal test app for the phone-detector ONNX model.
 *
 * The preprocessing / decode / NMS below is a faithful port of the Python
 * pipeline (train.py letterbox, model.py ExportWrapper + decode_all,
 * predict.py post-processing, evaluate.py nms):
 *
 *  - Letterbox: scale = min(320/w, 320/h); nw/nh = round(w*scale) (Python
 *    round, i.e. half-to-even); bilinear resize; pasted centred on a
 *    320x320 canvas filled with grey 114; dx = (320-nw)/2, dy = (320-nh)/2.
 *  - Input tensor: RGB / 255.0 -> [0,1], CHW, name "input". The (x-0.5)/0.5
 *    normalisation lives INSIDE the model graph.
 *  - Output "detections": [500, 5] = (x1, y1, x2, y2, score) in 320x320
 *    canvas coordinates, score already sigmoid(objectness).
 *  - Filter score >= 0.25, greedy NMS with IoU threshold 0.5, then map back
 *    with ox = clamp((x - dx) / scale, 0, w0) etc.
 */
class MainActivity : Activity() {

    companion object {
        private const val IMG_SIZE = 320
        private const val PAD_VALUE = 114
        private const val CONF_THRES = 0.25
        private const val NMS_IOU = 0.5
        private const val REQ_PICK = 1001
    }

    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var pickButton: Button

    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var session: OrtSession? = null
    @Volatile private var env: OrtEnvironment? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }
        pickButton = Button(this).apply {
            text = "Pick photo"
            setOnClickListener { pickPhoto() }
        }
        imageView = ImageView(this).apply {
            adjustViewBounds = true
            setBackgroundColor(Color.DKGRAY)
        }
        resultText = TextView(this).apply {
            textSize = 16f
            gravity = Gravity.START
            text = "Pick a photo to run the phone detector."
        }
        root.addView(pickButton)
        root.addView(
            imageView,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
        )
        root.addView(resultText)
        setContentView(root)
    }

    private fun pickPhoto() {
        val intent = Intent(
            Intent.ACTION_PICK,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        )
        @Suppress("DEPRECATION")
        startActivityForResult(intent, REQ_PICK)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_PICK || resultCode != RESULT_OK) return
        val uri: Uri = data?.data ?: return
        resultText.text = "Loading photo…"
        worker.execute {
            try {
                val bitmap = loadOrientedBitmap(uri)
                    ?: throw IllegalStateException("Could not decode the selected image.")
                runOnUiThread {
                    imageView.setImageBitmap(bitmap)
                    resultText.text = "Detecting…"
                }
                detect(bitmap)
            } catch (e: Exception) {
                runOnUiThread {
                    resultText.text = "Error: ${e.message ?: e.javaClass.simpleName}"
                }
            }
        }
    }

    /** Decode the picked image and apply its EXIF orientation. */
    private fun loadOrientedBitmap(uri: Uri): Bitmap? {
        val raw = contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it)
        } ?: return null

        var orientation = ExifInterface.ORIENTATION_NORMAL
        try {
            contentResolver.openInputStream(uri)?.use { input ->
                orientation = ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            }
        } catch (_: Exception) {
            // No readable EXIF: use the bitmap as decoded.
        }
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f); matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f); matrix.postScale(-1f, 1f)
            }
            else -> return raw
        }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
    }

    // ------------------------------------------------------------------
    // Faithful ports of the Python pipeline
    // ------------------------------------------------------------------

    /** Python's round(): round-half-to-even, for positive values. */
    private fun pyRound(v: Double): Int {
        val f = floor(v)
        val frac = v - f
        return when {
            frac > 0.5 -> f.toInt() + 1
            frac < 0.5 -> f.toInt()
            else -> if (f.toLong() % 2L == 0L) f.toInt() else f.toInt() + 1
        }
    }

    private data class Letterboxed(
        val tensor: FloatArray,
        val scale: Double,
        val dx: Int,
        val dy: Int
    )

    /** train.py letterbox() + predict.py's /255.0 conversion. */
    private fun letterbox(src: Bitmap): Letterboxed {
        val w = src.width
        val h = src.height
        val scale = min(IMG_SIZE.toDouble() / w, IMG_SIZE.toDouble() / h)
        val nw = max(1, pyRound(w * scale))
        val nh = max(1, pyRound(h * scale))
        val resized = Bitmap.createScaledBitmap(src, nw, nh, true)

        val canvasBmp = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        val dx = (IMG_SIZE - nw) / 2
        val dy = (IMG_SIZE - nh) / 2
        canvas.drawBitmap(resized, dx.toFloat(), dy.toFloat(), null)

        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        canvasBmp.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        val plane = IMG_SIZE * IMG_SIZE
        val tensor = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            tensor[i] = Color.red(p) / 255f
            tensor[plane + i] = Color.green(p) / 255f
            tensor[2 * plane + i] = Color.blue(p) / 255f
        }
        return Letterboxed(tensor, scale, dx, dy)
    }

    /** evaluate.py nms(): greedy NMS, keeps boxes with IoU <= threshold. */
    private fun nms(boxes: Array<DoubleArray>, scores: DoubleArray): List<Int> {
        if (boxes.isEmpty()) return emptyList()
        val n = boxes.size
        val areas = DoubleArray(n) {
            max(0.0, boxes[it][2] - boxes[it][0]) * max(0.0, boxes[it][3] - boxes[it][1])
        }
        var order: List<Int> = (0 until n).sortedByDescending { scores[it] }
        val keep = mutableListOf<Int>()
        while (order.isNotEmpty()) {
            val i = order[0]
            keep.add(i)
            if (order.size == 1) break
            val rest = order.subList(1, order.size)
            val next = ArrayList<Int>(rest.size)
            for (j in rest) {
                val xx1 = max(boxes[i][0], boxes[j][0])
                val yy1 = max(boxes[i][1], boxes[j][1])
                val xx2 = min(boxes[i][2], boxes[j][2])
                val yy2 = min(boxes[i][3], boxes[j][3])
                val inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                val union = areas[i] + areas[j] - inter
                val iou = inter / max(union, 1e-9)
                if (iou <= NMS_IOU) next.add(j)
            }
            order = next
        }
        return keep
    }

    private fun getSession(): OrtSession {
        session?.let { return it }
        val environment = OrtEnvironment.getEnvironment()
        env = environment
        val modelBytes = assets.open("best.onnx").readBytes()
        val created = environment.createSession(modelBytes)
        session = created
        return created
    }

    private data class Detection(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float
    )

    private fun detect(original: Bitmap) {
        val lb = letterbox(original)
        val sess = getSession()
        val environment = env ?: OrtEnvironment.getEnvironment()

        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(lb.tensor),
            longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        )
        val startNs = SystemClock.elapsedRealtimeNanos()
        val raw: FloatArray = inputTensor.use { tensor ->
            sess.run(mapOf("input" to tensor)).use { result ->
                val outTensor = result[0] as OnnxTensor
                val buf = outTensor.floatBuffer
                val arr = FloatArray(buf.remaining())
                buf.get(arr)
                arr
            }
        }
        val inferenceMs = (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0

        // Filter by confidence (predict.py), then NMS.
        val candBoxes = ArrayList<DoubleArray>()
        val candScores = ArrayList<Double>()
        var r = 0
        while (r + 4 < raw.size) {
            val score = raw[r + 4]
            if (score >= CONF_THRES) {
                candBoxes.add(
                    doubleArrayOf(
                        raw[r].toDouble(), raw[r + 1].toDouble(),
                        raw[r + 2].toDouble(), raw[r + 3].toDouble()
                    )
                )
                candScores.add(score.toDouble())
            }
            r += 5
        }
        val boxesArr = candBoxes.toTypedArray()
        val scoresArr = candScores.toDoubleArray()
        val keep = if (boxesArr.isNotEmpty()) nms(boxesArr, scoresArr) else emptyList()

        // Map back to original image coordinates (predict.py).
        val w0 = original.width.toDouble()
        val h0 = original.height.toDouble()
        val detections = keep.map { idx ->
            val b = boxesArr[idx]
            Detection(
                min(max((b[0] - lb.dx) / lb.scale, 0.0), w0).toFloat(),
                min(max((b[1] - lb.dy) / lb.scale, 0.0), h0).toFloat(),
                min(max((b[2] - lb.dx) / lb.scale, 0.0), w0).toFloat(),
                min(max((b[3] - lb.dy) / lb.scale, 0.0), h0).toFloat(),
                scoresArr[idx].toFloat()
            )
        }.sortedByDescending { it.score }

        val annotated = drawDetections(original, detections)
        runOnUiThread {
            imageView.setImageBitmap(annotated)
            resultText.text = buildString {
                append("Phones found: ").append(detections.size)
                append(String.format("  •  Inference: %.0f ms", inferenceMs))
                for (d in detections) {
                    append(String.format("\nphone %.2f", d.score))
                }
            }
        }
    }

    /** predict.py drawing: red rectangle (255,0,0) + "phone 0.96" label above. */
    private fun drawDetections(src: Bitmap, detections: List<Detection>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        if (detections.isEmpty()) return out
        val canvas = Canvas(out)
        // Python draws width=3 in original pixels; scale stroke/text a little
        // with image size so labels stay legible on high-res phone photos.
        val stroke = max(3f, src.width / 240f)
        val boxPaint = Paint().apply {
            color = Color.rgb(255, 0, 0)
            style = Paint.Style.STROKE
            strokeWidth = stroke
        }
        val textPaint = Paint().apply {
            color = Color.rgb(255, 0, 0)
            textSize = max(28f, src.width / 22f)
            isAntiAlias = true
        }
        for (d in detections) {
            canvas.drawRect(d.x1, d.y1, d.x2, d.y2, boxPaint)
            val label = String.format("phone %.2f", d.score)
            val ty = max(textPaint.textSize, d.y1 - 4f)
            canvas.drawText(label, d.x1 + 2f, ty, textPaint)
        }
        return out
    }

    override fun onDestroy() {
        worker.shutdownNow()
        try { session?.close() } catch (_: Exception) {}
        super.onDestroy()
    }
}
