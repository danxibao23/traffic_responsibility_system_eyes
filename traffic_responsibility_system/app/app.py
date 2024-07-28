from flask import Flask, request, jsonify
from infer import infer

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file.save('temp.wav')
        result = infer('temp.wav')
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
