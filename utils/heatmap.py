import cv2
import numpy as np
import tensorflow as tf
from keras import Model


def generate_heatmap(
    model: Model,
    decoded_image,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a heatmap overlay on the original image using Grad-CAM technique.

    Args:
        model (Model): The pre-trained model used for prediction.
        image_path (str): The path to the input image.
        last_conv_layer_name (str, optional): The name of the last convolutional layer in the
        model. If not provided, the last convolutional layer with "conv" in its name will be
        used.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the superimposed image with heatmap
        overlay and the prediction result of the model for the input image.
    """
    # image_plot = cv2.imread(image_path)  # pylint: disable=no-member
    # image_plot = cv2.cvtColor(  # pylint: disable=no-member
    #     image_plot, cv2.COLOR_BGR2RGB  # pylint: disable=no-member
    # )

    image_plot = cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR)

    resized_image = tf.image.resize(decoded_image, (224, 224))
    image_array = tf.expand_dims(resized_image, 0)

    prediction = model.predict(image_array, verbose=0)

    layer_names = [layer.name for layer in model.layers]
    last_conv_layer_name = [name for name in layer_names if "conv" in name][-1]

    gradcam_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradcam_model(image_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(  # pylint: disable=no-member
        heatmap, (decoded_image.shape[1], decoded_image.shape[0])
    )
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.cvtColor(  # pylint: disable=no-member
        cv2.applyColorMap(heatmap, cv2.COLORMAP_JET),  # pylint: disable=no-member
        cv2.COLOR_BGR2RGB,  # pylint: disable=no-member
    )

    ## Applying the heatmap to the original image
    superimposed_img = heatmap * 0.4 + image_plot
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    return superimposed_img, prediction
