
$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	
		
		let file = $("#image-selector").prop('files')[0];
		reader.readAsDataURL(file);

		setTimeout(simulateClick.bind(null,'predict-button'), 500);
});


let model;
(async function () {
	
	 model = await tf.loadModel('https://saketchaturvedi.github.io/tfjs-models/model.json');	
	 $("#selected-image").attr("src", "https://saketchaturvedi.github.io/assets/img.png")
	// Hide the model loading spinner
    $('.progress-bar').hide();
})();

$("#predict-button").click(async function () {
	
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
    .toFloat();

	
	
	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();
	
	
	
	
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	let predictions = await model.predict(tensor).data();
	let top7 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: SKIN_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 7);
	
	
    $("#prediction-list").empty();
    top7.forEach(function (p) {

	    $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);

	
	});	
});