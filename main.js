 async function predictCactus(img){
        const model = await tf.loadLayersModel('jsModel/model.json');
        let result = document.getElementById('isCactus');
        

        const prediction = model.predict(processImage(img)).dataSync();

        console.log(prediction);

        prediction > .5 ? result.textContent = 'IT IS A CACTUS!!!'
            : result.textContent = 'There aint no cactus in that picture!';

    }

    function processImage(img){
        let tensor = tf.browser.fromPixels(img);

        const resized = tf.image.resizeBilinear(tensor, [32,32]).toFloat();

        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));
        const batched = normalized.expandDims(0)

        return batched;
    }