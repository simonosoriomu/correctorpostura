// script.js
// Configuración global
const state = {
    images: {
        correct: [],
        forward: [],
        backward: []
    },
    isTraining: false,
    modelTrained: false,
    isPredicting: false,
    alertCounter: 0, 
    lastPrediction: null
};

// Clases
const CLASSES = ['correct', 'forward', 'backward'];
const CLASS_LABELS = ['Postura Correcta', 'Inclinado Adelante', 'Inclinado Atrás'];
const ALERT_THRESHOLD = 5; // Cantidad de frames seguidos con mala postura para lanzar alerta

// Referencias DOM
const fileInputs = {
    correct: document.getElementById('upload-correct'),
    forward: document.getElementById('upload-forward'),
    backward: document.getElementById('upload-backward')
};
const counts = {
    correct: document.getElementById('count-correct'),
    forward: document.getElementById('count-forward'),
    backward: document.getElementById('count-backward')
};
const thumbsContainers = {
    correct: document.getElementById('thumbs-correct'),
    forward: document.getElementById('thumbs-forward'),
    backward: document.getElementById('thumbs-backward')
};

const btnTrain = document.getElementById('btn-train');
const trainingStatus = document.getElementById('training-status');
const detectionSection = document.getElementById('detection-section');
const btnStartCamera = document.getElementById('btn-start-camera');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const webcamElement = document.getElementById('webcam');
const postureAlert = document.getElementById('posture-alert');
const postureConfidence = document.getElementById('posture-confidence');
const alertHeading = postureAlert.querySelector('h3');

// IA Models
let mobilenet;
let customModel;

// Inicialización de la aplicación
async function init() {
    trainingStatus.classList.remove('hidden');
    trainingStatus.innerText = "Cargando modelo base de IA...";
    
    try {
        // Cargamos MobileNet, lo usamos solo como extractor de características (truncado)
        const mobilenetRaw = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
        
        // Obtenemos una capa intermedia para usarla como salida de extracción de características
        const layer = mobilenetRaw.getLayer('conv_pw_13_relu');
        mobilenet = tf.model({inputs: mobilenetRaw.inputs, outputs: layer.output});
        
        trainingStatus.innerText = "IA lista. Por favor, sube imágenes para las tres posturas.";
    } catch (e) {
        trainingStatus.innerText = "Error cargando la IA. Revisa tu conexión a internet.";
        console.error(e);
    }
}

// Escuchar cargas de archivos
CLASSES.forEach(className => {
    fileInputs[className].addEventListener('change', (e) => handleImageUpload(e, className));
});

function handleImageUpload(event, className) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.className = 'thumb-img';
            imgElement.onload = () => {
                // Guardamos la imagen en nuestro estado
                state.images[className].push(imgElement);
                // Actualizamos UI
                counts[className].innerText = `${state.images[className].length} imágenes`;
                thumbsContainers[className].appendChild(imgElement);
            };
        };
        reader.readAsDataURL(file);
    }
}

// Botón de entrenar
btnTrain.addEventListener('click', trainModel);

async function trainModel() {
    if (!mobilenet) {
        alert("El modelo base aún no ha cargado. Espera un momento.");
        return;
    }
    
    if (state.images.correct.length === 0 || state.images.forward.length === 0 || state.images.backward.length === 0) {
        alert("Por favor, sube al menos una imagen para CADA postura antes de entrenar.");
        return;
    }

    state.isTraining = true;
    btnTrain.disabled = true;
    trainingStatus.innerText = "Procesando imágenes...";

    // 1. Extraer características de todas las imágenes
    const xs = [];
    const ys = [];

    // Pequeño retardo para que la UI se actualice
    await new Promise(r => setTimeout(r, 100));

    // Procesamos cada clase
    for (let i = 0; i < CLASSES.length; i++) {
        const className = CLASSES[i];
        const images = state.images[className];
        
        for (let j = 0; j < images.length; j++) {
            const img = images[j];
            // Convertimos la imagen HTML a un Tensor y la preprocesamos para MobileNet
            const tensorImg = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .expandDims();
            // Normalizar de 0-255 a -1 a 1
            const normalizedImg = tensorImg.div(127.5).sub(1);
            
            // Pasamos la imagen por mobilenet para sacar características
            const activation = mobilenet.predict(normalizedImg);
            
            xs.push(activation);
            // Label es el índice de la clase (0, 1 o 2)
            ys.push(i);
            
            // Liberar tensores intermedios
            tf.dispose([tensorImg, normalizedImg]);
        }
    }

    // Unimos todos los tensores
    const xDataset = tf.concat(xs);
    const yDataset = tf.oneHot(tf.tensor1d(ys, 'int32'), 3);
    
    // Limpiamos los tensores individuales ya que los unimos
    xs.forEach(t => t.dispose());

    // 2. Construir el modelo clasificador personalizado
    customModel = tf.sequential({
        layers: [
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({
                units: 100,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            tf.layers.dense({
                units: 3, // 3 Clases
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    });

    customModel.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // 3. Entrenar
    trainingStatus.innerText = "Entrenando modelo (0%)...";
    
    const batchSize = Math.floor(xDataset.shape[0] * 0.2) > 0 ? Math.floor(xDataset.shape[0] * 0.2) : 1;
    const epochs = 20;
    
    await customModel.fit(xDataset, yDataset, {
        batchSize,
        epochs: epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const acc = (logs.acc * 100).toFixed(1);
                trainingStatus.innerText = `Entrenando: Época ${epoch + 1}/${epochs}. Precisión: ${acc}%`;
            }
        }
    });

    // Limpieza
    xDataset.dispose();
    yDataset.dispose();

    state.isTraining = false;
    state.modelTrained = true;
    
    trainingStatus.innerText = "¡Modelo entrenado correctamente!";
    trainingStatus.style.color = "var(--success-color)";
    
    // Habilitar sección de detección
    detectionSection.classList.remove('disabled');
}


// --- SECCIÓN DE CÁMARA Y DETECCIÓN ---

btnStartCamera.addEventListener('click', async () => {
    if (!state.modelTrained) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream;
        webcamElement.classList.remove('hidden');
        cameraPlaceholder.classList.add('hidden');
        
        state.isPredicting = true;
        // Iniciar loop de predicción una vez que el video pueda reproducirse
        webcamElement.addEventListener('loadeddata', predictLoop);
        
    } catch (e) {
        console.error(e);
        alert("No se pudo acceder a la cámara. Asegúrate de dar permisos.");
    }
});

async function predictLoop() {
    if (!state.isPredicting) return;
    
    // 1. Capturar frame de la webcam
    // tf.tidy limpia automáticamente la memoria de los tensores intermedios creados aquí
    tf.tidy(() => {
        // Convertir frame de video a tensor
        const img = tf.browser.fromPixels(webcamElement);
        // Recortar al centro para hacerlo cuadrado (opcional pero recomendado) y redimensionar
        // Para simplificar, solo redimensionamos a 224x224
        const resized = tf.image.resizeBilinear(img, [224, 224]);
        // Normalizar
        const normalized = resized.div(127.5).sub(1).expandDims(0);
        
        // 2. Extraer características y predecir
        const activation = mobilenet.predict(normalized);
        const predictions = customModel.predict(activation);
        
        // 3. Obtener el índice con mayor probabilidad
        const predictedClass = predictions.argMax(1).dataSync()[0];
        const confidence = predictions.max().dataSync()[0] * 100;
        
        updateUI(predictedClass, confidence.toFixed(1));
    });
    
    // Llamar al próximo frame
    requestAnimationFrame(predictLoop);
}

function updateUI(predictedClass, confidence) {
    const classStr = CLASSES[predictedClass];
    
    // Lógica de suavizado (debouncing)
    // Si la predicción es mala postura
    if (predictedClass === 1 || predictedClass === 2) {
        state.alertCounter++;
    } else {
        // Si es correcta, reseteamos rápido el contador para quitar la alerta pronto
        state.alertCounter = 0;
    }
    
    // Actualizar confianza siempre
    postureConfidence.innerText = `Confianza: ${confidence}%`;
    
    // Si el contador supera el umbral, mostramos alerta
    if (state.alertCounter > ALERT_THRESHOLD) {
        postureAlert.className = 'alert-box warning';
        alertHeading.innerText = `⚠️ CORRIGE TU POSTURA\n(${CLASS_LABELS[predictedClass]})`;
    } 
    // Si el contador es 0, mostramos correcta
    else if (state.alertCounter === 0) {
        postureAlert.className = 'alert-box success';
        alertHeading.innerText = "✓ Buena postura";
    }
    // Si está entre 0 y el umbral, dejamos el estado anterior para evitar parpadeo
}

// --- LÓGICA DE CONSENTIMIENTO ---
const consentModal = document.getElementById('consent-modal');
const btnAcceptConsent = document.getElementById('btn-accept-consent');
const btnDeclineConsent = document.getElementById('btn-decline-consent');
const consentActions = document.getElementById('consent-actions');
const consentDeniedMessage = document.getElementById('consent-denied-message');

// Bloquear scroll inicial
document.body.classList.add('modal-open');

btnAcceptConsent.addEventListener('click', () => {
    // Ocultar modal y permitir scroll
    consentModal.classList.add('hidden');
    document.body.classList.remove('modal-open');
    
    // Iniciar aplicación cargando TF solo DESPUÉS de que el usuario haya aceptado
    init();
});

btnDeclineConsent.addEventListener('click', () => {
    // Ocultar botones y mostrar mensaje de error
    consentActions.classList.add('hidden');
    consentDeniedMessage.classList.remove('hidden');
});
