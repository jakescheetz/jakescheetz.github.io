import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// Setup

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector('#bg'),
});

renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.setZ(30);
camera.position.setX(-3);

renderer.render(scene, camera);


// Lights

const pointLight = new THREE.PointLight(0xffffff);
pointLight.position.set(5, 5, 5);

const ambientLight = new THREE.AmbientLight(0xffffff);
scene.add(pointLight, ambientLight);


// function to add stars =====================
function addStar() {
  const geometry = new THREE.SphereGeometry(0.25, 24, 24);
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const star = new THREE.Mesh(geometry, material);

  const [x, y, z] = Array(3)
    .fill()
    .map(() => THREE.MathUtils.randFloatSpread(100));

  star.position.set(x, y, z);
  scene.add(star);
}

Array(200).fill().forEach(addStar);
//end randomly adding stars

// Background

const spaceTexture = new THREE.TextureLoader().load('space.jpg');
scene.background = spaceTexture;

// Avatar ========================================================
const jeffTexture = new THREE.TextureLoader().load('jake-removed.png');

const jeff = new THREE.Mesh(new THREE.BoxGeometry(3, 3, 3), new THREE.MeshBasicMaterial({ map: jeffTexture }));

scene.add(jeff);
jeff.position.z = -5;
jeff.position.x = 2;
// add avatar

// Moon ====================================================
const moonTexture = new THREE.TextureLoader().load('moon.jpg');
const normalTexture = new THREE.TextureLoader().load('normal.jpg');

const moon = new THREE.Mesh(
  new THREE.SphereGeometry(3, 32, 32),
  new THREE.MeshStandardMaterial({
    map: moonTexture,
    normalMap: normalTexture,
  })
);

scene.add(moon);

moon.position.z = 30;
moon.position.setX(-10);
//end moon ==================



// Earth =====================================================
const earthTexture = new THREE.TextureLoader().load('earth.png');

const earth = new THREE.Mesh(
  new THREE.SphereGeometry(2, 35, 35),
  new THREE.MeshStandardMaterial({
    map: earthTexture,
    normalMap: normalTexture,
  })
);

scene.add(earth);

earth.position.x = -5;
earth.position.z = 20;
//end earth ===============

// Sun ====================================================
const sunTexture = new THREE.TextureLoader().load('sun.jpg');

const sun = new THREE.Mesh(
  new THREE.SphereGeometry(9, 85, 85),
  new THREE.MeshStandardMaterial({
    map: sunTexture,
    normalMap: normalTexture,
  })
);

scene.add(sun);

sun.position.x = 15;
sun.position.z = 4;
sun.position.y = 5;
//end sun ==================

// Mars =====================================================
const marsTexture = new THREE.TextureLoader().load('mars.jpg');

const mars = new THREE.Mesh(
  new THREE.SphereGeometry(4, 65, 65),
  new THREE.MeshStandardMaterial({
    map: marsTexture,
    normalMap: normalTexture,
  })
);

scene.add(mars);

mars.position.x = -12;
mars.position.z = 50;
//end mars ==========================

// Saturn =====================================================
const saturnTexture = new THREE.TextureLoader().load('Saturn.jpg');

const saturn = new THREE.Mesh(
  new THREE.SphereGeometry(6, 85, 85),
  new THREE.MeshStandardMaterial({
    map: saturnTexture
  })
);

scene.add(saturn);

saturn.position.x = -15;
saturn.position.z = 80;

const saturnringTexture = new THREE.TextureLoader().load('saturnrings.jpg');

const saturnring = new THREE.Mesh(
  new THREE.TorusGeometry( 10, 1, 2.5, 100 ),
  new THREE.MeshStandardMaterial({
    map: saturnringTexture
  })
);

scene.add(saturnring);
saturnring.position.x = -15;
saturnring.position.z = 80;
saturnring.rotateX(90);

//end Saturn ==========================


// Scroll Animation

function moveCamera() {
  const t = document.body.getBoundingClientRect().top;
  moon.rotation.x += 0.01;
  moon.rotation.y += 0.025;
  moon.rotation.z += 0.01;

  earth.rotation.x += 0.01;
  earth.rotation.y += 0.025;
  earth.rotation.z += 0.01;

  mars.rotation.x += 0.01;
  mars.rotation.y += 0.025;
  mars.rotation.z += 0.01;

  sun.rotation.x += 0.001;
  sun.rotation.y += 0.002;
  sun.rotation.z += 0.001;

  saturn.rotation.x += 0.001;
  saturn.rotation.y += 0.001;
  saturn.rotation.z += 0.001;

  jeff.rotation.y += 0.005;
  jeff.rotation.z += 0.005;

  camera.position.z = t * -0.01;
  camera.position.x = t * -0.0002;
  camera.rotation.y = t * -0.0002;
}

document.body.onscroll = moveCamera;
moveCamera();

// Animation Loop

function animate() {
  requestAnimationFrame(animate);

  // torus.rotation.x += 0.01;
  // torus.rotation.y += 0.005;
  // torus.rotation.z += 0.01;

  moon.rotation.x += 0.005;
  earth.rotation.x += 0.005;
  mars.rotation.x += 0.005;
  sun.rotation.x += 0.001;
  saturn.rotation.x += 0.001;

  // controls.update();

  renderer.render(scene, camera);
}

animate();
