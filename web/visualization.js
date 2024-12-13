import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class Visualizer {
    constructor(node, container, visualSrc) {
        this.node = node;
        this.container = container;

        // Create main container div
        this.mainContainer = document.createElement('div');
        this.mainContainer.style.position = 'relative';
        this.mainContainer.style.width = '100%';
        this.mainContainer.style.height = '100%';
        container.appendChild(this.mainContainer);

        // Setup Three.js scene
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, this.mainContainer.clientWidth / this.mainContainer.clientHeight, 0.1, 1000);
        
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.mainContainer.clientWidth, this.mainContainer.clientHeight);
        this.mainContainer.appendChild(this.renderer.domElement);

        // Setup lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Setup camera and controls
        this.camera.position.z = 5;
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Start animation loop
        this.animate();

        // Handle window resize
        window.addEventListener('resize', () => this.onResize());
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const width = this.mainContainer.clientWidth;
        const height = this.mainContainer.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    updateVisual(params) {
        if (params?.filename) {
            // Load the GLB model
            const loader = new GLTFLoader();
            loader.load(params.filename, (gltf) => {
                // Clear existing model
                while(this.scene.children.length > 0) { 
                    const obj = this.scene.children[0];
                    if (obj.type === "Mesh") {
                        obj.geometry.dispose();
                        obj.material.dispose();
                    }
                    this.scene.remove(obj);
                }

                // Add new model
                this.scene.add(gltf.scene);

                // Center and scale model
                const box = new THREE.Box3().setFromObject(gltf.scene);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2.0 / maxDim;
                gltf.scene.scale.multiplyScalar(scale);
                
                gltf.scene.position.sub(center.multiplyScalar(scale));
            });
        }
    }

    remove() {
        // Clean up Three.js resources
        this.renderer.dispose();
        this.controls.dispose();
        this.container.remove();
    }
}

// Update the onExecuted handler in registerVisualizer
nodeType.prototype.onExecuted = async function(message) {
    const params = {};
    
    // Handle mesh/3D model
    if (message?.mesh) {
        params.filename = message.mesh[0];
    }
    
    this.updateParameters(params);
};

// Add this to the createVisualizer function
node.onResize = function() {
    let [w, h] = this.size;
    const minWidth = 400;
    const minHeight = 300;
    
    // Enforce minimum dimensions
    w = Math.max(w, minWidth);
    h = Math.max(h, minHeight);
    
    // Maintain aspect ratio
    if (w > minWidth) {
        h = Math.max(minHeight, w * 0.75); // 4:3 aspect ratio
    }
    
    this.size = [w, h];
    
    // Update visualizer size if it exists
    if (this.visualizer) {
        this.visualizer.onResize();
    }
}; 