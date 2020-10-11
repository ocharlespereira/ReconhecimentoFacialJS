const cam = document.getElementById('cam');

//verifica cameras instaladas
const startVideo = () => {
  navigator.mediaDevices.enumerateDevices().then((devices) => {
    if (Array.isArray(devices)) {
      devices.forEach((device) => {
        if (device.kind === 'videoinput') {
          // seleciona webCam correta
          if (device.label.includes('')) {
            navigator.getUserMedia(
              {
                video: {
                  deviceId: device.deviceId,
                },
              },
              (stream) => (cam.srcObject = stream),
              (error) => console.log(error)
            );
          }
        }
      });
    }
  });
}

//função para fazer o sistema reconhecer quem sou eu
const loadLabels = () => {
  const labels = ['Charles Pereira'] //define pasta onde irá procurar

  return Promise.all(labels.map(async (label) => {

    const descriptions = []

    for(let i = 1; i <= 4; i++) {
     const img = await faceapi.fetchImage(`/assets/img/${labels}/${i}.jpg`)
    
     const detections = await faceapi
      .detectSingleFace(img) //percorre img a img e verifica a que se melhor adequa
      .withFaceLandmarks() //detecta e reconhece os traços do rosto
      .withFaceDescriptor() //detecta a descricao do rosto
    
      descriptions.push(detections.descriptor)
    }

    return new faceapi.LabeledFaceDescriptors(label, descriptions) //retorna nova descrição por rotulo
  }))
}

Promise.all([  
  faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api/models'), // Detecta rostos do video, desenha o quadrado do rosto
  faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api/models'), // Desenha os traços do rosto, olhos, boca, nariz
  faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api/models'), //Usado para fazer o reconhecimento do rosto, quem sou eu, se me conhece ou não
  faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api/models'), //Detecta expressões se estou feliz, triste, ou bravo
  faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api/models'), // Detecta idade e genero e sexo
  faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api/models'), // Usada internamente p/ detectar rostos, desenhar o quadrado em tela
]).then(startVideo)

cam.addEventListener('play', async() => {
  const canvas = faceapi.createCanvasFromMedia(cam)
  const canvasSize = {
    width: cam.width,
    height: cam.height,
  }

  const labels = await loadLabels() //chama função para verificar imagem

  faceapi.matchDimensions(canvas, canvasSize)
  document.body.appendChild(canvas)
  
  setInterval(async () => {
    //desenha o quadrado em volta do rosto com probabilidade de acerto
    const detections = await faceapi
      .detectAllFaces(
        cam, 
        new faceapi.TinyFaceDetectorOptions()
      )
      .withFaceLandmarks() //detecta e reconhece os traços do rosto
      .withFaceExpressions() //detecta expressoes faciais
      .withAgeAndGender() //detecta idade e genero
      .withFaceDescriptors() //detecta a descricao do rosto
    
    const resizedDetections = faceapi.resizeResults(detections, canvasSize)

    const faceMatcher = new faceapi.FaceMatcher(labels, 0.6) //taxa de acerto de comparação de img com o BD
    const results = resizedDetections.map(d => 
      faceMatcher.findBestMatch(d.descriptor)
    )  //procura o melhor resultado

    canvas.getContext('2d').clearRect(0,0, canvas.width, canvas.height)
    faceapi.draw.drawDetections(canvas, resizedDetections)

    //desenhar os traços do rosto
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)

    //desenha expressoes faciais
    faceapi.draw.drawFaceExpressions(canvas,  resizedDetections)

    //descobre idade e genero
    resizedDetections.forEach(detection => {
      const {age, gender, genderProbability} = detection;
      new faceapi.draw.DrawTextField([
        `${parseInt(age,10)} - years`,
        `${gender} (${parseInt(genderProbability * 100,10)}%)`
      ], detection.detection.box.topRight).draw(canvas)
    })

    results.forEach((result, index) => {
      const box = resizedDetections[index].detection.box
      const {label, distance} = result
      new faceapi.draw.DrawTextField([
        `${label} (${parseInt(distance * 100,10)}%)`,
      ], box.bottomRight).draw(canvas)
    })
  },100)
})