from transformers.convert_graph_to_onnx import convert
from onnxconverter_common import auto_convert_mixed_precision_model_path
import onnx
import torch.onnx
import onnxruntime


def process_predictions_ans(flattened_preds, threshold=0.95):
    """
    Processes predictions by applying a threshold to distinguish between a specific class and others.
    It assumes softmax has already been applied to the predictions.

    Parameters:
    - flattened_preds: A list of prediction tensors, with each tensor representing predictions for a batch.
    - threshold: A probability threshold used to decide whether to classify a prediction as a specific class or as 'other'.

    Returns:
    - preds_final: A list of numpy arrays with final predictions after applying the threshold.
    """

    print("\nPrediction")
    preds_final = []  # Initialize a list to store final predictions

    # Iterate over each set of predictions
    for predictions in flattened_preds:
        # softmax was applied to the first dimension before averaging
        predictions_softmax = predictions

        # Get the argmax across all classes
        predictions_argmax = predictions.argmax(-1)

        # Get predictions for all classes except 'O'
        predictions_without_O = predictions_softmax[:, :12].argmax(-1)

        # Get the softmax probabilities for the 'O' class
        O_predictions = predictions_softmax[:, 12]

        # Apply threshold to decide between 'O' class and other classes
        pred_final = torch.where(
            O_predictions < threshold,
            predictions_without_O,
            predictions_argmax)

        # Convert final predictions to numpy array and add to the list
        preds_final.append(pred_final.numpy())

    return preds_final


def predict_and_convert(data_loader, model, config, onnx_model_path):
    """
    Exports the given model to the ONNX format after processing a single batch from the data loader.

    Parameters:
    - data_loader: DataLoader object to provide input data for the model.
    - model: The model to be exported to ONNX format.
    - config: Configuration object containing device and others
    - onnx_model_path: Path where the ONNX model will be saved.

    Returns:
    - prediction_outputs: List of model outputs for the processed batch. Currently initialized but not used.
    """

    # Set the model to evaluation mode
    model.eval()

    # Initialize a list to store the prediction outputs
    prediction_outputs = []

    # Create an iterator from the DataLoader
    data_iter = iter(data_loader)

    # Fetch the first batch of data from the iterator
    batch = next(data_iter)

    # Disable gradient calculations for export
    with torch.no_grad():

        # Prepare inputs by reshaping and moving them to the specified device
        inputs = {
            key: val.reshape(
                val.shape[0], -1).to(
                config.device) for key, val in batch.items() if key in [
                'input_ids', 'attention_mask']}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Export the model to ONNX format with the specified configurations
        torch.onnx.export(model,  # Model to be exported
                          # Example model input
                          args=(input_ids, attention_mask),
                          f=onnx_model_path,  # Path to save the ONNX model
                          opset_version=12,  # ONNX opset version
                          # Names of the input parameters
                          input_names=['input_ids', 'attention_mask'],
                          output_names=['logits'],  # Names of the output
                          dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},  # Dynamic axes for batching
                                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'}}
                          )

    print("Model saved to", onnx_model_path)

    return prediction_outputs


def predict_and_quant(
        data_loader,
        config,
        original_onnx_model_path,
        output_file_name,
        data_path):
    """
    Performs quantization on a given ONNX model based on a single batch from the data loader and saves the quantized model.

    Parameters:
    - data_loader: DataLoader object providing input data for quantization.
    - config: Configuration object containing device settings.
    - original_onnx_model_path: Path to the original ONNX model that will be quantized.
    - output_file_name: Filename for the quantized ONNX model.
    - data_path: Path where additional data related to quantization might be stored.

    Returns:
    - prediction_outputs: List of model outputs for the processed batch. Currently, it only appends a placeholder value.
    """

    # Initialize a list to store prediction outputs
    prediction_outputs = []

    # Create an iterator from the DataLoader
    data_iter = iter(data_loader)

    # Fetch the first batch of data from the iterator
    batch = next(data_iter)

    # Disable gradient calculations for efficiency
    with torch.no_grad():

        # Prepare inputs by reshaping and moving them to the specified device
        inputs = {
            key: val.reshape(
                val.shape[0], -1).to(
                config.device) for key, val in batch.items() if key in [
                'input_ids', 'attention_mask']}

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Quantization process
        print("Quantization")

        # Prepare input data for quantization by moving tensors to CPU and
        # converting to numpy arrays
        input_data = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()}

        # Call the function to auto convert the ONNX model to mixed precision
        # with specified settings
        auto_convert_mixed_precision_model_path(
            original_onnx_model_path,  # Original ONNX model path
            input_data,  # Input data for calibration during quantization
            output_file_name,  # Output file name for the quantized model
            # Specify the execution provider, can be changed to CPU if
            # necessary
            provider=['CUDAExecutionProvider'],
            location=data_path,  # specify the path to save external data tensors
            rtol=2,  # Relative tolerance for quantization
            atol=20,  # Absolute tolerance for quantization
            keep_io_types=True,  # Maintain input/output types
            verbose=True  # Enable verbose output during quantization
        )

        # Append a placeholder value to prediction outputs (currently not used
        # for actual predictions)
        prediction_outputs.append(0)

    print("Model saved to", output_file_name)

    return prediction_outputs


def predict(data_loader, session, config):
    """
    Performs inference using a given ONNX model session over all batches from a data loader.

    Parameters:
    - data_loader: DataLoader object providing batches of input data for inference.
    - session: The ONNX runtime session initialized with the model to be used for inference.
    - config: Configuration object containing settings

    Returns:
    - processed_predictions: List of processed predictions after inference for all input data.
    """

    # Initialize a list to collect raw predictions for each batch
    prediction_outputs = []

    # Iterate over all batches of data from the data loader
    for batch in tqdm(data_loader, desc="Predicting"):
        with torch.no_grad():
            # Prepare inputs by reshaping and moving them to the specified
            # device
            inputs = {
                key: val.reshape(
                    val.shape[0], -1).to(
                    config.device) for key, val in batch.items() if key in [
                    'input_ids', 'attention_mask']}

            # Retrieve the names of the input and output nodes from the model
            # session
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]

            # Extract input_ids and attention_mask from inputs
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Prepare input data by moving tensors to CPU and converting to
            # numpy arrays
            input_data = {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy()}

            # Execute the model
            onnx_outputs = session.run(None, input_data)

            # Append raw model outputs (predictions) to the list
            # Assuming the first output is what we need
            prediction_outputs.append(torch.tensor(onnx_outputs[0]))

    # Flatten the list of predictions across all batches
    prediction_outputs = [
        logit for batch in prediction_outputs for logit in batch]

    # Process the predictions as required (e.g., applying softmax,
    # thresholding)
    processed_predictions = process_predictions(prediction_outputs)

    return processed_predictions
