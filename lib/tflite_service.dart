import 'dart:developer';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteService {
  Interpreter? _interpreter;
  List<String>? _labels;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'leaf_disease_efficientnetb0.tflite',
        options: InterpreterOptions(),
      );

      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData.split('\n');

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      log('Input tensor: shape=${inputTensor.shape}, type=${inputTensor.type}', name: 'TFLiteService');
      log('Output tensor: shape=${outputTensor.shape}, type=${outputTensor.type}', name: 'TFLiteService');

      log('Model and labels loaded successfully', name: 'TFLiteService');
    } catch (e) {
      log('Error loading model: $e', name: 'TFLiteService', level: 1000);
    }
  }

  Future<Map<String, double>?> runInferenceOnBytes(Uint8List imageBytes) async {
    if (_interpreter == null || _labels == null) {
      log('Model not loaded', name: 'TFLiteService', level: 900);
      return null;
    }

    final input = _preprocessImage(imageBytes);
    if (input == null) return null;

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final output = List.generate(outputShape[0], (_) => List.filled(outputShape[1], 0.0));

    try {
      _interpreter!.run([input], output);
    } catch (e) {
      log('Error running inference: $e', name: 'TFLiteService', level: 1000);
      return null;
    }

    final results = output[0];
    final topResult = _getTopResult(results);

    return topResult;
  }

  List<List<List<num>>>? _preprocessImage(Uint8List imageBytes) {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) return null;

      final resizedImage = img.copyResize(image, width: 224, height: 224);

      var imageMatrix = List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resizedImage.getPixel(x, y);
            return [pixel.r, pixel.g, pixel.b];
          },
        ),
      );

      return imageMatrix;
    } catch (e) {
      log('Error preprocessing image: $e', name: 'TFLiteService', level: 1000);
      return null;
    }
  }

  Map<String, double> _getTopResult(List<double> results) {
    double maxScore = -1;
    int maxIndex = -1;

    for (int i = 0; i < results.length; i++) {
      if (results[i] > maxScore) {
        maxScore = results[i];
        maxIndex = i;
      }
    }

    if (maxIndex != -1 && _labels != null && maxIndex < _labels!.length) {
      final label = _labels![maxIndex].trim();
      return {label: maxScore};
    }

    return {};
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
