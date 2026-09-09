from flask import Flask, render_template, request, jsonify
from prediction_model import predictor
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main virtual DCB testing interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle virtual DCB prediction requests:
    - Parses and strictly validates physical input properties
    - Evaluates the zero-leakage trained ExtraTreesRegressor model
    - Computes ASTM D5528 physics-consistent compliance and SERR curves
    - Returns single authoritative prediction curve, graph arrays, and OOD warnings
    """
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        required_fields = [
            'E11', 'E22', 'G12', 'poisson_ratio', 'ply_thickness',
            'loading_rate', 'width', 'thickness', 'initial_crack_avg'
        ]

        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f"Missing required field: '{field}'"}), 400
            if input_data[field] is None or input_data[field] == '':
                return jsonify({'error': f"Field '{field}' cannot be empty"}), 400

        # Convert to float
        parsed_inputs = {}
        for field in required_fields:
            try:
                parsed_inputs[field] = float(input_data[field])
            except (ValueError, TypeError):
                return jsonify({'error': f"Field '{field}' must be a valid numeric value"}), 400

        # Run authoritative prediction pipeline
        res = predictor.predict(parsed_inputs)

        return jsonify({
            'predictions': res['predictions'],
            'critical_point': res['critical_point'],
            'prediction_curve': res['prediction_curve'],
            'graphs': res['graphs'],
            'input_domain_status': res['input_domain_status'],
            'warnings': res['warnings'],
            'is_ood': res['is_ood'],
            'model_info': res['model_info'],
            'message': 'Prediction completed successfully'
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f"Internal prediction failure: {str(e)}"}), 500

@app.route('/model-info')
def model_info():
    """Return model information and training evaluation metrics"""
    return jsonify(predictor.get_model_info_summary())

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'scaler_loaded': predictor.scaler is not None
    })

if __name__ == '__main__':
    print("Starting AI-Assisted Virtual DCB Testing Platform...")
    print("Access the dashboard at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
