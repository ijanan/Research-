# Blueprint: Leaf Disease Detector

## 1. Overview
A Flutter mobile application to classify the condition of plant leaves using a pre-trained TensorFlow Lite (TFLite) model. The app will determine if a leaf is healthy or identify the specific type of disease.

## 2. Core Features
-   **Image Input:**
    -   **Live Camera Scan:** Use the device's camera for real-time leaf analysis.
    -   **Gallery Upload:** Select an existing photo from the device's gallery.
-   **Analysis & Display:**
    -   Display the selected image.
    -   Show the top prediction label (e.g., "Potato___Late_blight").
    -   Display the confidence score of the prediction.

## 3. Architecture
-   **UI (View):** A clean, intuitive interface built with Flutter widgets.
-   **State Management:** Use `ValueNotifier` for simple, local state management.
-   **Service Layer:** A `TFLiteService` to handle all logic related to loading the model and running inference. This separates the ML logic from the UI code.

## 4. Current Plan
-   [x] **Project Initialization & Blueprint:** Create this `blueprint.md` file.
-   [ ] **Dependency Integration:** Add `image_picker` and `tflite_flutter` to `pubspec.yaml`.
-   [ ] **Asset Setup:** Create an `assets` folder and configure `pubspec.yaml`. **User action required: You will need to upload your `.tflite` model and `.txt` labels file to this new `assets` folder.**
-   [ ] **UI Scaffolding:** Implement the basic home screen with buttons for camera/gallery access and placeholders for the image and results.
