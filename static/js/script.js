function predict() {
    var formData = new FormData();
    var imageFile = document.getElementById('image-upload').files[0];
    formData.append('image', imageFile);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}