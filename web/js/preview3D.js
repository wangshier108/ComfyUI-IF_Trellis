// web/js/preview3D.js

import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

// Import Three.js from CDN
const THREE = await import("https://unpkg.com/three@0.157.0/build/three.module.js");
const { GLTFLoader } = await import("https://unpkg.com/three@0.157.0/examples/jsm/loaders/GLTFLoader.js");
const { OrbitControls } = await import("https://unpkg.com/three@0.157.0/examples/jsm/controls/OrbitControls.js");

const html_path = "/extensions/ComfyUI-IF_Trellis/html/preview3D.html";

app.registerExtension({
    name: "Comfy.IF_Preview3D",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "IF_Preview_3D") return;

        // Setup Three.js viewer
        class ModelViewer {
            constructor(container) {
                this.container = container;
                this.scene = new THREE.Scene();
                this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                
                this.renderer = new THREE.WebGLRenderer({ 
                    antialias: true,
                    alpha: true 
                });
                this.renderer.setClearColor(0x000000, 0);
                this.renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(this.renderer.domElement);
                
                // Add lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                this.scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                directionalLight.position.set(1, 1, 1);
                this.scene.add(directionalLight);
                
                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.target.set(0, 0, 0);
                
                this.camera.position.z = 5;
                
                this.animate = this.animate.bind(this);
                this.loader = new GLTFLoader();
                
                window.addEventListener('resize', () => this.onResize());
                this.animate();
            }
            
            loadModel(url) {
                // Show loading indicator
                if (!this.loadingText) {
                    this.loadingText = document.createElement('div');
                    this.loadingText.style.position = 'absolute';
                    this.loadingText.style.top = '50%';
                    this.loadingText.style.left = '50%';
                    this.loadingText.style.transform = 'translate(-50%, -50%)';
                    this.loadingText.style.color = 'white';
                    this.container.appendChild(this.loadingText);
                }
                this.loadingText.textContent = 'Loading model...';
                
                // Clear existing model
                while(this.scene.children.length > 0) { 
                    const obj = this.scene.children[0];
                    if (obj.type === "Mesh") {
                        obj.geometry.dispose();
                        obj.material.dispose();
                    }
                    this.scene.remove(obj);
                }
                
                // Load new model
                this.loader.load(
                    url,
                    (gltf) => {
                        this.scene.add(gltf.scene);
                        
                        // Center and scale model
                        const box = new THREE.Box3().setFromObject(gltf.scene);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());
                        
                        const maxDim = Math.max(size.x, size.y, size.z);
                        const scale = 2.0 / maxDim;
                        gltf.scene.scale.multiplyScalar(scale);
                        
                        gltf.scene.position.sub(center.multiplyScalar(scale));
                        
                        // Hide loading text
                        if (this.loadingText) {
                            this.loadingText.style.display = 'none';
                        }
                    },
                    (xhr) => {
                        const percent = (xhr.loaded / xhr.total * 100).toFixed(0);
                        if (this.loadingText) {
                            this.loadingText.textContent = `Loading model... ${percent}%`;
                        }
                    },
                    (error) => {
                        console.error('Error loading model:', error);
                        if (this.loadingText) {
                            this.loadingText.textContent = 'Error loading model';
                        }
                    }
                );
            }
            
            animate() {
                requestAnimationFrame(this.animate);
                this.controls.update();
                this.renderer.render(this.scene, this.camera);
            }
            
            onResize() {
                const width = this.container.clientWidth;
                const height = this.container.clientHeight;
                
                this.camera.aspect = width / height;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(width, height);
            }
        }

        // Add preview container and setup
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            const paths = message.ui;
            
            // Create/update preview container
            if (!this.preview_container) {
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "400px";
                container.style.position = "relative";
                this.preview_container = container;
                this.content.appendChild(container);
                
                // Create model viewer
                this.viewer = new ModelViewer(container);
            }

            // Update model if path provided
            if (paths.glb) {
                this.viewer.loadModel(paths.glb);
            }
        };

        // Add menu options
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }

            options.push(
                {
                    content: "Open in New Window",
                    callback: () => {
                        window.open(html_path + "?glb=" + encodeURIComponent(this.widgets[0].value), "_blank");
                    }
                },
                {
                    content: "Reset View",
                    callback: () => {
                        if (this.viewer) {
                            this.viewer.controls.reset();
                        }
                    }
                }
            );
        };
    }
});