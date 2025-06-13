from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from src.adcp_analysis import ADCPAnalyzer
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'log', 'dat'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use .txt, .log, or .dat files'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"📁 Saving file to: {filepath}")
        file.save(filepath)
        
        # Check file size and first few lines for debugging
        file_size = os.path.getsize(filepath)
        print(f"📊 File size: {file_size} bytes")
        
        # Read first few lines to check content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
                print(f"📄 First 5 lines:")
                for i, line in enumerate(first_lines, 1):
                    print(f"   {i}: {repr(line)}")
        except UnicodeDecodeError:
            print("⚠️  File might have encoding issues, trying with different encoding...")
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    print(f"📄 First 5 lines (latin-1):")
                    for i, line in enumerate(first_lines, 1):
                        print(f"   {i}: {repr(line)}")
            except Exception as e:
                print(f"❌ Could not read file with any encoding: {e}")
        
        # Initialize analyzer and process the file
        print("🔬 Starting analysis...")
        analyzer = ADCPAnalyzer()
        
        try:
            results = analyzer.analyze_file(filepath)
            print("✅ Analysis completed successfully")
        except Exception as analysis_error:
            print(f"❌ Analysis failed: {str(analysis_error)}")
            print(f"📍 Full traceback:")
            traceback.print_exc()
            
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'error': f'Analysis failed: {str(analysis_error)}',
                'detailed_error': traceback.format_exc(),
                'file_info': {
                    'size': file_size,
                    'name': filename
                }
            }), 500
        
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            print("🗑️  Temporary file cleaned up")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'results': results
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"💥 Unexpected error: {error_trace}")
        
        # Try to clean up file if it exists
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'detailed_error': error_trace
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🌊 Starting ADCP Analysis Web App with debugging enabled...")
    app.run(debug=True, host='0.0.0.0', port=8003)
