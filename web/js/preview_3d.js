// preview_3d.js
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.Preview3D",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Preview3D") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                this.setupPreviewElement();
            };
            
            nodeType.prototype.setupPreviewElement = function() {
                // Create preview container
                const elem = document.createElement("div");
                elem.style.width = "100%";
                elem.style.height = "200px";
                elem.style.backgroundColor = "#333";
                this.preview = elem;
                
                // Add to node
                this.widgets_start = this.widgets_start || [];
                this.widgets_start.push(elem);
            };
            
            // Handle model updates
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                if (message?.ui?.model_file?.[0]) {
                    const modelPath = message.ui.model_file[0];
                    // Update preview with new model
                    this.loadModel(modelPath);
                }
            };
        }
    }
});