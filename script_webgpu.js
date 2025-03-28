document.addEventListener('DOMContentLoaded', () => {
    const trackCanvas = document.getElementById('track-canvas');
    const trackCtx = trackCanvas.getContext('2d');
    const carCanvas = document.getElementById('car-canvas');
    const carCtx = carCanvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const testBtn = document.getElementById('test-btn');
    const statusText = document.getElementById('track-status');
    const canvasContainer = document.querySelector('.canvas-container');
    const carInstructions = document.querySelector('.car-instructions');
    const distanceCounter = document.getElementById('distance-counter');
    const finishNotification = document.getElementById('finish-notification');
    const runRLBtn = document.getElementById('run-rl-btn');
    const stopRLBtn = document.getElementById('force-rl-btn');
    const exportModelBtn = document.getElementById('export-model-btn');
    const importModelBtn = document.getElementById('import-model-btn');
    const modelFileInput = document.getElementById('model-file-input');
    // Network visualization elements
    const networkVisualization = document.getElementById('network-visualization');
    const networkCanvas = document.getElementById('network-canvas');
    const networkCtx = networkCanvas.getContext('2d');
    const activationStats = document.getElementById('activation-stats');
    const nodesStats = document.getElementById('nodes-stats');
    const connectionsStats = document.getElementById('connections-stats');
    let generationCount = 1;
    let isTesting = false;
    let crashedAgentsHistory = [];
    // Current best network for visualization
    let bestNetwork = null;

    const exportTrackBtn = document.getElementById('export-track-btn'); // New
    const importTrackBtn = document.getElementById('import-track-btn'); // New
    const trackFileInput = document.getElementById('track-file-input'); // New


    // Track constants
    const startPointRadius = 10;
    const pointConnectionThreshold = 20; // Distance in pixels to consider connected to start
    const trackWidth = 50; // Increased track width to fit car
    const startPointColor = '#27ae60';
    const trackColor = '#333';
    const finishPointColor = '#e74c3c';

    // Lidar constants
    const LIDAR_RANGE = 110; // Length of lidar rays
    const LIDAR_COUNT = 24; // Number of lidar rays
    const LIDAR_COLOR_SAFE = '#3498db'; // Blue for lines inside track
    const LIDAR_COLOR_DANGER = '#e74c3c'; // Red for lines outside track

    // NodeGene represents a neuron in the network.
    // It can be an 'input', 'output', 'hidden', or 'bias' node.
    class NodeGene {
        constructor(id, type, activationType = 'sigmoid') {
            this.id = id;
            this.type = type; // 'input', 'output', 'hidden', or 'bias'
            this.activationType = activationType; // 'sigmoid', 'relu', 'tanh', 'linear'
            this.value = 0;
        }

        // Apply the appropriate activation function based on type
        activate(x) {
            switch (this.activationType) {
                case 'sigmoid':
                    return 1 / (1 + Math.exp(-x));
                case 'relu':
                    return Math.max(0, x);
                case 'leaky_relu':
                    return x > 0 ? x : 0.01 * x;
                case 'tanh':
                    return Math.tanh(x);
                case 'linear':
                    return x;
                default:
                    return 1 / (1 + Math.exp(-x)); // Default to sigmoid
            }
        }
        static activationTypeMap = {
            'sigmoid': 0,
            'relu': 1,
            'tanh': 2,
            'linear': 3,
            'leaky_relu': 4,
            // Add more as needed
        };

        // Static map for node types to integer IDs for GPU
        static nodeTypeMap = {
            'input': 0,
            'bias': 1,
            'hidden': 2,
            'output': 3,
        };
    }

    const neatFeedForwardWGSL = `
    struct NodeInfo { nodeType: u32, activationType: u32, };
    // Ensure Connection struct matches JS packing (u32, u32, f32, u32)
    struct Connection { inNodeId: u32, outNodeId: u32, weight: f32, enabled: u32, };
    struct Params { maxNodeId: u32, numConnections: u32, };

    // Input buffers
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage, read> nodeInfos: array<NodeInfo>;
    @group(0) @binding(2) var<storage, read> connections: array<Connection>;
    @group(0) @binding(3) var<storage, read> valuesIn: array<f32>;

    // Output buffer
    @group(0) @binding(4) var<storage, read_write> valuesOut: array<f32>;

    // Activation functions
    fn activate(activationType: u32, x: f32) -> f32 {
        switch activationType {
            case 0u: { // sigmoid
                return 1.0 / (1.0 + exp(-x));
            }
            case 1u: { // relu
                return max(0.0, x);
            }
            case 2u: { // tanh
                let ex = exp(x);
                let enx = exp(-x);
                return (ex - enx) / (ex + enx);
            }
            case 3u: { // linear
                return x;
            }
            case 4u: { // leaky_relu
                return select(0.01 * x, x, x > 0.0);
            }
            default: { // Default sigmoid
                return 1.0 / (1.0 + exp(-x));
            }
        }
    }

    @compute @workgroup_size(64) // Ensure this matches workgroupSize in JS dispatch
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let nodeId = global_id.x;
        // Boundary check: Don't process if nodeId is out of bounds
        if (nodeId >= params.maxNodeId) {
            return;
        }

        let info = nodeInfos[nodeId];
        let nodeType = info.nodeType;

        // Input (0) and Bias (1) nodes: Output value is just their input value
        if (nodeType == 0u || nodeType == 1u) {
            valuesOut[nodeId] = valuesIn[nodeId];
            return; // Done processing this node
        }

        // Hidden (2) and Output (3) nodes: Calculate weighted sum
        var sum: f32 = 0.0;

        // Loop through ALL connections to find those feeding into the current nodeId
        // The loop limit MUST be the actual number of connections passed in params
        for (var i: u32 = 0u; i < params.numConnections; i = i + 1u) {
            let conn = connections[i];

            // Check if this connection's output is the current node AND it's enabled
            if (conn.outNodeId == nodeId && conn.enabled != 0u) { // Compare enabled (u32) with 0u
                let inId = conn.inNodeId;
                // Safety check: ensure the input node ID is valid before accessing valuesIn
                if (inId < params.maxNodeId) {
                   sum = sum + valuesIn[inId] * conn.weight;
                }
                // Else: Connection references an invalid input node? Skip it.
            }
        }

        // Apply the activation function AFTER summing all relevant inputs
        valuesOut[nodeId] = activate(info.activationType, sum);

    } // End of main function
`; // End of WGSL string

    // ConnectionGene represents a weighted connection between two nodes.
    class ConnectionGene {
        constructor(inNode, outNode, weight, enabled, innovation) {
            this.inNode = inNode;   // Source node id
            this.outNode = outNode; // Target node id
            this.weight = weight;
            this.enabled = enabled;
            this.innovation = innovation;
        }
    }

    // NeuralNetwork implements a simple NEAT-style network.
    // It supports feedforward computation and mutations (weight, add connection, add node).
    class NeuralNetwork {
        constructor(inputSize, outputSize) {
            this.nodes = new Map(); // Map of node id to NodeGene
            this.connections = [];  // Array of ConnectionGene
            this.nextNodeId = 0;    // Unique id for the next node gene
            this.nextInnovationNumber = 0; // Unique innovation number for new connections

            // Add properties for WebGPU state
            this.gpuDevice = null; // Store the GPUDevice
            this.gpuComputePipeline = null;
            this.gpuNodeInfosBuffer = null;
            this.gpuConnectionsBuffer = null;
            this.gpuParamsBuffer = null;
            this.gpuBindGroupLayout = null;

            // Create input nodes.
            for (let i = 0; i < inputSize; i++) {
                const node = new NodeGene(this.nextNodeId++, 'input');
                this.nodes.set(node.id, node);
            }

            // Create a bias node.
            const bias = new NodeGene(this.nextNodeId++, 'bias');
            this.nodes.set(bias.id, bias);

            // Create hidden layer with a few nodes (new addition)
            const hiddenLayerSize = 8;
            const hiddenNodes = [];
            const hiddenActivationTypes = ['tanh', 'relu', 'leaky_relu']; // Or just one type initially
            for (let i = 0; i < hiddenLayerSize; i++) {
                const randomActivation = hiddenActivationTypes[Math.floor(Math.random() * hiddenActivationTypes.length)];
                // const node = new NodeGene(this.nextNodeId++, 'hidden', 'linear'); // OLD Linear
                const node = new NodeGene(this.nextNodeId++, 'hidden', randomActivation); // NEW Random/Default
                this.nodes.set(node.id, node);
                hiddenNodes.push(node);
            }
            // Ensure output nodes start as linear
             const outputNodes = [];
             for (let i = 0; i < outputSize; i++) {
                 const node = new NodeGene(this.nextNodeId++, 'output', 'linear'); // Keep linear
                 this.nodes.set(node.id, node);
                 outputNodes.push(node);
             }

            // Fully connect input and bias nodes to hidden nodes
            const inputAndBiasNodes = [...this.nodes.values()].filter(n => n.type === 'input' || n.type === 'bias');
            for (let inNode of inputAndBiasNodes) {
                for (let hiddenNode of hiddenNodes) {
                    this.addConnection(inNode.id, hiddenNode.id, Math.random() * 2 - 1);
                }
            }

            // Fully connect hidden nodes to output nodes
            for (let hiddenNode of hiddenNodes) {
                for (let outNode of outputNodes) {
                    this.addConnection(hiddenNode.id, outNode.id, Math.random() * 2 - 1);
                }
            }
        }

        // Adds a connection between two nodes with a given weight.
        // Prevents duplicate connections.
        addConnection(inNodeId, outNodeId, weight, enabled = true) {
            if (this.connections.find(conn => conn.inNode === inNodeId && conn.outNode === outNodeId)) {
                return; // Skip duplicate
            }
            const connection = new ConnectionGene(
                inNodeId,
                outNodeId,
                weight,
                enabled,
                this.nextInnovationNumber++
            );
            this.connections.push(connection);
        }

        // Computes the network output for a given array of input values.
        // It uses a simple iterative algorithm to ensure that each node is processed once its inputs are ready.
        feedForward(inputArray) {
            // ... (same as original) ...
            // Reset all node values.
            for (let node of this.nodes.values()) node.value = 0;
            // Set input node values.
            const inputNodes = [...this.nodes.values()].filter(n => n.type === 'input');
            if (inputArray.length !== inputNodes.length) throw new Error("Input array length does not match number of input nodes.");
            for (let i = 0; i < inputNodes.length; i++) inputNodes[i].value = inputArray[i];
            // Set bias node value.
            const biasNodes = [...this.nodes.values()].filter(n => n.type === 'bias');
            for (let b of biasNodes) b.value = 1;
            // Process remaining nodes.
            const remaining = new Set([...this.nodes.keys()].filter(id => this.nodes.get(id).type !== 'input' && this.nodes.get(id).type !== 'bias'));
            let maxIterations = this.nodes.size * 2; // Safety break for potential cycles
            while (remaining.size > 0 && maxIterations-- > 0) {
                let processedThisIteration = false;
                for (let nodeId of Array.from(remaining)) {
                    const node = this.nodes.get(nodeId);
                    const incoming = this.connections.filter(conn => conn.outNode === nodeId && conn.enabled);
                    const ready = incoming.every(conn => !remaining.has(conn.inNode));
                    if (ready) {
                        let sum = 0;
                        for (let conn of incoming) {
                            sum += this.nodes.get(conn.inNode).value * conn.weight;
                        }
                        node.value = node.activate(sum);
                        remaining.delete(nodeId);
                        processedThisIteration = true;
                    }
                }
                if (!processedThisIteration && remaining.size > 0) {
                    console.warn("FeedForward: Possible cycle or disconnected node detected. Breaking early.");
                    break; // Break if no progress is made
                }
            }
            if (maxIterations <= 0) {
                console.warn("FeedForward: Max iterations reached. Network might have cycles or be complex.");
            }
            // Collect output node values.
            const outputNodes = [...this.nodes.values()].filter(n => n.type === 'output');
            return outputNodes.map(n => n.value);
        }

        // Mutates connection weights.
        // With a given probability, each connection is either perturbed or assigned a new random weight.
        mutateWeights(probability = 0.8, perturbChance = 0.9, stepSize = 0.1) {
            for (let conn of this.connections) {
                if (Math.random() < probability) {
                    if (Math.random() < perturbChance) {
                        // Perturb the weight slightly.
                        conn.weight += (Math.random() * 2 - 1) * stepSize;
                    } else {
                        // Assign a completely new random weight.
                        conn.weight = Math.random() * 2 - 1;
                    }
                }
            }
        }

        // Attempts to add a new connection between two randomly selected nodes.
        // Skips invalid selections (e.g., connections from output/bias to input).
        mutateAddConnection(maxTries = 100) {
            const nodeIds = Array.from(this.nodes.keys());
            for (let i = 0; i < maxTries; i++) {
                const inNodeId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
                const outNodeId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
                const inNode = this.nodes.get(inNodeId);
                const outNode = this.nodes.get(outNodeId);

                // Avoid connections from output or bias to input.
                if (inNode.type === 'output' || inNode.type === 'bias') continue;
                if (outNode.type === 'input' || outNode.type === 'bias') continue;
                // Prevent duplicate connections.
                if (this.connections.find(conn => conn.inNode === inNodeId && conn.outNode === outNodeId)) continue;

                // (For simplicity, cycle detection is not implemented here.)
                this.addConnection(inNodeId, outNodeId, Math.random() * 2 - 1);
                return;
            }
        }

        // Adds a new node by splitting an existing enabled connection.
        // The old connection is disabled and replaced by two new connections.
        mutateAddNode() {
            // Select a random enabled connection.
            const enabledConnections = this.connections.filter(conn => conn.enabled);
            if (enabledConnections.length === 0) return;
            const conn = enabledConnections[Math.floor(Math.random() * enabledConnections.length)];
            conn.enabled = false;

            // Create a new hidden node.
            const newNode = new NodeGene(this.nextNodeId++, 'hidden');
            this.nodes.set(newNode.id, newNode);

            // Create a connection from the original inNode to the new node (weight = 1).
            this.addConnection(conn.inNode, newNode.id, 1.0);
            // Create a connection from the new node to the original outNode (weight copied from the disabled connection).
            this.addConnection(newNode.id, conn.outNode, conn.weight);
        }

        // Add this to the NeuralNetwork class
        // Add this to the NeuralNetwork class
        mutateActivations(nodeMutationProbability = 0.1) { // Renamed param for clarity, reasonable default
            const activationTypes = ['sigmoid', 'relu', 'leaky_relu', 'tanh', 'linear'];

            for (let [nodeId, node] of this.nodes.entries()) {
                // Only mutate hidden and output nodes (input/bias usually fixed)
                // *** IMPORTANT: Decide if output nodes should remain linear or can mutate ***
                // Option 1: Allow output nodes to mutate (except if you need specific output ranges)
                // if (node.type === 'hidden' || node.type === 'output') {

                // Option 2: Only mutate hidden nodes, keep output linear (safer for control tasks)
                 if (node.type === 'hidden') {

                    if (Math.random() < nodeMutationProbability) { // Use the passed probability
                        // Choose a random activation function that's different from current
                        let newType;
                        if (activationTypes.length <= 1) continue; // Cannot change if only one type exists

                        do {
                            newType = activationTypes[Math.floor(Math.random() * activationTypes.length)];
                        } while (newType === node.activationType); // Loop until a *different* type is found

                        // console.log(`Mutating activation for node ${nodeId} from ${node.activationType} to ${newType}`);
                        node.activationType = newType;
                    }
                }
                 // Ensure output nodes ALWAYS remain linear if that's required for the car controls
                 else if (node.type === 'output' && node.activationType !== 'linear') {
                    // console.warn(`Correcting output node ${nodeId} activation to linear.`);
                    node.activationType = 'linear';
                 }
            }
        }

        // Performs crossover between two parent networks.
        // This method assumes both parents have the same input/output sizes.
        // Matching genes are randomly chosen from either parent while disjoint/excess genes are inherited from parent1.
        // Performs crossover between two parent networks.
        // Ensures all nodes referenced by inherited connections are present in the child.
        static crossover(parent1, parent2) {
            // Create a child network with no initial nodes/connections.
            const child = new NeuralNetwork(0, 0);
            child.nodes = new Map();
            child.connections = [];

            // --- Updated Node Handling ---
            // 1. Add all nodes from parent1 to the child.
            for (let [id, node] of parent1.nodes.entries()) {
                // Create a new NodeGene instance for the child
                child.nodes.set(id, new NodeGene(node.id, node.type, node.activationType));
            }

            // 2. Add nodes from parent2 that are not already in the child.
            //    Also, handle potential activation type conflicts for existing nodes.
            for (let [id, node2] of parent2.nodes.entries()) {
                if (child.nodes.has(id)) {
                    const existingNode = child.nodes.get(id);
                     // *** Keep linear if forced, otherwise maybe randomize (original logic was ok) ***
                     if (existingNode.type === 'hidden' && existingNode.activationType !== 'linear') {
                         if (Math.random() < 0.5) { // Original logic was random choice
                            existingNode.activationType = node2.activationType;
                         }
                         // Ensure it doesn't accidentally become non-linear if we are forcing linear
                         // This might be overly complex, disabling mutation is easier for the test.
                     }
                    // Ensure output nodes remain linear
                    if (existingNode.type === 'output') {
                        existingNode.activationType = 'linear';
                    }
                } else {
                     // Node only exists in parent2, add it to child.
                     const newNode = new NodeGene(node2.id, node2.type, node2.activationType);
                     // *** Force linear if testing ***
                     if (newNode.type === 'hidden' || newNode.type === 'output') {
                         newNode.activationType = 'linear';
                     }
                     child.nodes.set(id, newNode);
                }
            }

            for (let node of child.nodes.values()) {
                if (node.type === 'output') {
                     node.activationType = 'linear';
                }
                // *** Force hidden to linear FOR TEST ***
                if (node.type === 'hidden') {
                     node.activationType = 'linear';
                }
            }
            // --- End Updated Node Handling ---


            // Map parent2's connections by their innovation number for efficient lookup.
            const parent2Conns = new Map();
            for (let conn of parent2.connections) {
                parent2Conns.set(conn.innovation, conn);
            }

            // Process each connection gene from parent1.
            for (let conn1 of parent1.connections) {
                let conn2 = parent2Conns.get(conn1.innovation);
                let chosenConn = null;

                if (conn2 !== undefined) {
                    // Matching gene: randomly choose between parent1's or parent2's connection.
                    chosenConn = Math.random() < 0.5 ? conn1 : conn2;

                    // Ensure the chosen connection is enabled if EITHER parent had it enabled (common practice).
                    const enabled = conn1.enabled || conn2.enabled;

                    // If randomly disabled during crossover (optional, low probability)
                    const disableChance = 0.25; // e.g., 25% chance to disable an inherited gene if both parents had it
                    const finalEnabled = (Math.random() > disableChance) ? enabled : false;

                    // Create the child connection using the chosen weight/nodes but potentially re-enabled status.
                    child.connections.push(new ConnectionGene(
                        chosenConn.inNode,
                        chosenConn.outNode,
                        chosenConn.weight,
                        finalEnabled, // Use the potentially re-enabled status
                        chosenConn.innovation
                    ));

                    // Remove from parent2 map to handle excess genes later (though not strictly needed with this loop structure)
                    parent2Conns.delete(conn1.innovation);

                } else {
                    // Disjoint or excess gene from parent1: inherit directly.
                    chosenConn = conn1;
                    child.connections.push(new ConnectionGene(
                        chosenConn.inNode,
                        chosenConn.outNode,
                        chosenConn.weight,
                        chosenConn.enabled,
                        chosenConn.innovation
                    ));
                }

                // --- Sanity Check (Optional but Recommended) ---
                // Verify that the nodes for the chosen connection actually exist in the child.
                if (!child.nodes.has(chosenConn.inNode) || !child.nodes.has(chosenConn.outNode)) {
                    console.error("Crossover Error: Child network missing node referenced by chosen connection!",
                        "Connection:", chosenConn,
                        "Child Nodes:", Array.from(child.nodes.keys()));
                    // Depending on desired robustness, you might skip adding this connection,
                    // throw an error, or try to add the missing node (though the node handling above should prevent this).
                }
                // --- End Sanity Check ---
            }

            // Note: We don't need to explicitly handle excess genes from parent2 here,
            // because the standard NEAT crossover typically only inherits excess/disjoint
            // genes from the fitter parent (assumed to be parent1 here).

            // Update next IDs, taking the max from both parents.
            child.nextNodeId = Math.max(parent1.nextNodeId, parent2.nextNodeId);
            child.nextInnovationNumber = Math.max(parent1.nextInnovationNumber, parent2.nextInnovationNumber);

            return child;
        }

        async initGPU() {
            if (this.gpuDevice) return true; // Already initialized

            if (!navigator.gpu) {
                console.error("WebGPU not supported on this browser.");
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error("No appropriate GPUAdapter found.");
                return false;
            }

            this.gpuDevice = await adapter.requestDevice();
            if (!this.gpuDevice) {
                console.error("Failed to get GPUDevice.");
                return false;
            }

            console.log("WebGPU Device Initialized:", this.gpuDevice);

            // --- Create reusable GPU resources ---


            const device = this.gpuDevice;

            // Create Shader Module
            const shaderModule = device.createShaderModule({ code: neatFeedForwardWGSL });

            // Create Bind Group Layout (defines structure for bindings)
            this.gpuBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },    // params
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // nodeInfos
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // connections
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // valuesIn
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }       // valuesOut
                ]
            });

            // Create Compute Pipeline
            try {
                device.pushErrorScope('validation');
                this.gpuComputePipeline = device.createComputePipeline({
                    layout: device.createPipelineLayout({
                        bindGroupLayouts: [this.gpuBindGroupLayout]
                    }),
                    compute: {
                        module: shaderModule,
                        entryPoint: "main"
                    }
                });
                const pipelineError = await device.popErrorScope();
                if (pipelineError) {
                    console.error("Pipeline Creation Error:", pipelineError);
                    throw new Error(`Pipeline creation failed: ${pipelineError.message}`);
                }
            } catch (err) {
                console.error("Error during pipeline creation block:", err);
                return false; // Indicate failure
            }
            // --- Create static buffers (node info, connections, params) ---
            // These buffers might need updating if the network structure changes (mutations)

            const maxNodeId = this.nextNodeId; // Use nextNodeId as upper bound
            const numConnections = this.connections.length;

            // 1. Node Infos Buffer (nodeType, activationType)
            const nodeInfoData = new Uint32Array(maxNodeId * 2); // 2 uint32s per node
            for (const [id, node] of this.nodes.entries()) {
                if (id < maxNodeId) { // Ensure ID is within bounds
                    nodeInfoData[id * 2 + 0] = NodeGene.nodeTypeMap[node.type] ?? NodeGene.nodeTypeMap['hidden']; // Default if type unknown
                    nodeInfoData[id * 2 + 1] = NodeGene.activationTypeMap[node.activationType] ?? NodeGene.activationTypeMap['sigmoid']; // Default
                }
            }
            this.gpuNodeInfosBuffer = device.createBuffer({
                size: nodeInfoData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true, // Map on creation for initial data write
            });
            new Uint32Array(this.gpuNodeInfosBuffer.getMappedRange()).set(nodeInfoData);
            this.gpuNodeInfosBuffer.unmap();


            // 2. Connections Buffer (inId, outId, weight, enabled)
            // Using 4 floats per connection for simplicity (alignment)
            const connectionData = new Float32Array(numConnections * 4);
            for (let i = 0; i < numConnections; i++) {
                const conn = this.connections[i];
                connectionData[i * 4 + 0] = conn.inNode;
                connectionData[i * 4 + 1] = conn.outNode;
                connectionData[i * 4 + 2] = conn.weight;
                connectionData[i * 4 + 3] = conn.enabled ? 1.0 : 0.0;
            }

            const outputNodeIds = Array.from(this.nodes.values())
                .filter(n => n.type === 'output')
                .map(n => n.id);
            console.log("Output Node IDs:", outputNodeIds);
            let loggedOutputWeights = 0;
            for (let i = 0; i < numConnections && loggedOutputWeights < 10; i++) { // Log first few
                const conn = this.connections[i];
                if (outputNodeIds.includes(conn.outNode)) {
                    console.log(`Conn to Output ${conn.outNode} (from ${conn.inNode}): Weight=${connectionData[i * 4 + 2].toFixed(4)}, Enabled=${connectionData[i * 4 + 3]}`);
                    loggedOutputWeights++;
                }
            }

            console.log(`initGPU - Creating Params Buffer: maxNodeId=${maxNodeId}, numConnections=${numConnections}`);
            if (numConnections === 0 && this.nodes.size > (LIDAR_COUNT + 1 + 4)) { // Only warn if connections expected
                console.warn("initGPU - !!! Initial number of connections is zero. Shader loop might not run correctly. !!!");
            }
            // Check bias connections specifically
            const biasNodeId = Array.from(this.nodes.values()).find(n => n.type === 'bias')?.id;
            if (biasNodeId !== undefined) {
                console.log(`Bias Node ID: ${biasNodeId}`);
                for (let i = 0; i < numConnections; i++) {
                    const conn = this.connections[i];
                    if (conn.inNode === biasNodeId && outputNodeIds.includes(conn.outNode)) {
                        console.log(`Bias Conn to Output ${conn.outNode}: Weight=${connectionData[i * 4 + 2].toFixed(4)}, Enabled=${connectionData[i * 4 + 3]}`);
                    }
                }
            }

            const hiddenNodeIds = Array.from(this.nodes.values())
                .filter(n => n.type === 'hidden')
                .map(n => n.id);
            console.log("Hidden Node IDs:", hiddenNodeIds);
            if (biasNodeId !== undefined) {
                console.log(`-- Connections FROM Bias Node (${biasNodeId}) --`);
                for (let i = 0; i < numConnections; i++) {
                    const conn = this.connections[i];
                    if (conn.inNode === biasNodeId && hiddenNodeIds.includes(conn.outNode)) {
                        console.log(`Bias Conn to Hidden ${conn.outNode}: Weight=${connectionData[i * 4 + 2].toFixed(4)}, Enabled=${connectionData[i * 4 + 3]}`);
                    }
                }
            }
            this.gpuConnectionsBuffer = device.createBuffer({
                size: connectionData.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            new Float32Array(this.gpuConnectionsBuffer.getMappedRange()).set(connectionData);
            this.gpuConnectionsBuffer.unmap();

            // 3. Params Buffer (maxNodeId, numConnections)
            const paramsData = new Uint32Array([maxNodeId, numConnections]);
            this.gpuParamsBuffer = device.createBuffer({
                // Uniform buffers need size aligned to 16 bytes
                size: Math.max(16, paramsData.byteLength),
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            if (!this.gpuParamsBuffer) {
                console.error("initGPU: Failed to create Params buffer.");
                this.gpuNodeInfosBuffer?.destroy(); // Clean up previously created buffers
                this.gpuConnectionsBuffer?.destroy();
                return false;
            }
            new Uint32Array(this.gpuParamsBuffer.getMappedRange()).set(paramsData);
            this.gpuParamsBuffer.unmap();


            console.log("GPU Resources Initialized (Pipeline, Static Buffers)");
            return true;
        }

        // Call this after mutations that change network structure (add node/connection)
        updateGPUStructureBuffers() {
            if (!this.gpuDevice) return; // Only if GPU is initialized
            console.log("Updating GPU structure buffers...");

            // --- Destroy old buffers before creating new ones ---
            this.gpuNodeInfosBuffer?.destroy();
            this.gpuConnectionsBuffer?.destroy();
            this.gpuParamsBuffer?.destroy();
            this.gpuNodeInfosBuffer = null;
            this.gpuConnectionsBuffer = null;
            this.gpuParamsBuffer = null;


            const device = this.gpuDevice;
            const maxNodeId = this.nextNodeId;
            const numConnections = this.connections.length;

            console.log(`  - updateGPUStructureBuffers - Updating Params: maxNodeId=${maxNodeId}, numConnections=${numConnections}`);
            if (numConnections === 0 && this.nodes.size > (LIDAR_COUNT + 1 + 4)) { // Check if connections expected
                console.warn("  - updateGPUStructureBuffers - !!! Number of connections is zero. Shader loop might not run correctly. !!!");
            }

            // 1. Recreate Node Infos Buffer
            const nodeInfoData = new Uint32Array(maxNodeId * 2);
            for (const [id, node] of this.nodes.entries()) {
                if (id < maxNodeId) {
                    nodeInfoData[id * 2 + 0] = NodeGene.nodeTypeMap[node.type] ?? NodeGene.nodeTypeMap['hidden'];
                    nodeInfoData[id * 2 + 1] = NodeGene.activationTypeMap[node.activationType] ?? NodeGene.activationTypeMap['sigmoid'];
                }
            }
            try {
                this.gpuNodeInfosBuffer = device.createBuffer({
                    size: nodeInfoData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
                });
                new Uint32Array(this.gpuNodeInfosBuffer.getMappedRange()).set(nodeInfoData);
                this.gpuNodeInfosBuffer.unmap();
            } catch (err) {
                console.error("Error creating Node Infos buffer:", err);
                return false; // Indicate failure
            }

            // 2. Recreate Connections Buffer
            const bytesPerConnection = 16; // u32(4) + u32(4) + f32(4) + u32(4)
            const connectionBufferSize = numConnections * bytesPerConnection;

            console.log(`initGPU - Preparing Connections Buffer: numConnections=${numConnections}, bufferSize=${connectionBufferSize} bytes`);
            if (numConnections === 0 && this.nodes.size > (LIDAR_COUNT + 1 + 4)) { // Use LIDAR_COUNT etc.
                console.warn("initGPU - !!! Initial number of connections is zero. Shader loop might not run correctly. !!!");
            }

            // Create ArrayBuffer and DataView
            const connectionBufferAB = new ArrayBuffer(connectionBufferSize);
            const connectionDataView = new DataView(connectionBufferAB);
            const littleEndian = true; // Assuming little endian

            // Populate the ArrayBuffer using DataView
            for (let i = 0; i < numConnections; i++) {
                const conn = this.connections[i];
                const offset = i * bytesPerConnection;

                // Write data with correct types
                connectionDataView.setUint32(offset + 0, conn.inNode, littleEndian);
                connectionDataView.setUint32(offset + 4, conn.outNode, littleEndian);
                connectionDataView.setFloat32(offset + 8, conn.weight, littleEndian);
                connectionDataView.setUint32(offset + 12, conn.enabled ? 1 : 0, littleEndian); // Write enabled as u32 (1 or 0)

                // Optional: Log first few connections during initGPU to verify data before buffer creation
                if (i < 5) {
                     const logIn = connectionDataView.getUint32(offset + 0, littleEndian);
                     const logOut = connectionDataView.getUint32(offset + 4, littleEndian);
                     const logW = connectionDataView.getFloat32(offset + 8, littleEndian);
                     const logEn = connectionDataView.getUint32(offset + 12, littleEndian);
                     console.log(`  initGPU DataView Conn ${i}: in=${logIn}, out=${logOut}, w=${logW.toFixed(4)}, en=${logEn}`);
                }
            }

            // Create the GPU buffer
            try {
                this.gpuConnectionsBuffer = device.createBuffer({
                    size: connectionBufferSize, // Use calculated size
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    mappedAtCreation: true,
                });
                // Copy the raw ArrayBuffer data into the GPU buffer
                new Uint8Array(this.gpuConnectionsBuffer.getMappedRange()).set(new Uint8Array(connectionBufferAB));
                this.gpuConnectionsBuffer.unmap();
            } catch(err) {
                 console.error("Error creating initial Connections buffer:", err);
                 // Clean up other buffers if creation fails
                 this.gpuNodeInfosBuffer?.destroy();
                 this.gpuParamsBuffer?.destroy();
                 this.gpuConnectionsBuffer = null; // Ensure it's null on failure
                 return false; // Indicate failure
            }


            // 3. Recreate Params Buffer
            const paramsData = new Uint32Array([maxNodeId, numConnections]);
            try {
                this.gpuParamsBuffer = device.createBuffer({
                    size: Math.max(16, paramsData.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
                });
                new Uint32Array(this.gpuParamsBuffer.getMappedRange()).set(paramsData);
                this.gpuParamsBuffer.unmap();
            } catch (err) {
                console.error("Error creating Params buffer:", err);
                this.gpuNodeInfosBuffer?.destroy();
                this.gpuConnectionsBuffer?.destroy();
                return;
            }
        }


        // --- WebGPU FeedForward ---
        async feedForwardGPU(inputArray, maxIterations = 0) {
            if (!this.gpuDevice || !this.gpuComputePipeline || !this.gpuNodeInfosBuffer || !this.gpuConnectionsBuffer || !this.gpuParamsBuffer) {
                console.log("GPU resources not ready, attempting initialization...");
                const success = await this.initGPU();
                if (!success) {
                    throw new Error("WebGPU initialization failed. Cannot perform feedforward.");
                }
                // Check again after init attempt
                if (!this.gpuDevice || !this.gpuComputePipeline || !this.gpuNodeInfosBuffer || !this.gpuConnectionsBuffer || !this.gpuParamsBuffer) {
                    throw new Error("GPU resource initialization incomplete after attempt.");
                }
            }

            const device = this.gpuDevice;
            const maxNodeId = this.nextNodeId; // Current max ID + 1 is buffer size needed

            // Determine number of iterations
            // For acyclic graphs, depth is roughly <= node count. Add buffer.
            let iterations = 1000; // Try 3x nodes, or even more (e.g., 1000 fixed?)


            // --- Prepare Initial Node Values ---
            const initialValues = new Float32Array(maxNodeId).fill(0); // Initialize all to 0

            const inputNodes = [...this.nodes.values()].filter(n => n.type === 'input');
            if (inputArray.length !== inputNodes.length) {
                throw new Error("Input array length does not match number of input nodes.");
            }
            // Set input values
            inputNodes.forEach((node, i) => {
                if (node.id < maxNodeId) initialValues[node.id] = inputArray[i];
            });
            // Set bias values
            this.nodes.forEach(node => {
                if (node.type === 'bias' && node.id < maxNodeId) initialValues[node.id] = 1.0;
            });


            // --- Create Per-Invocation GPU Buffers ---
            const bufferSize = maxNodeId * Float32Array.BYTES_PER_ELEMENT;

            console.log("Initial Values:", initialValues.slice(0, inputNodes.length + 5)); // Log inputs, bias, maybe a few hidden

            // 1. Values Buffer In (ping)
            const gpuValuesBufferIn = device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, // Need COPY_SRC if iterations is odd
            });
            device.queue.writeBuffer(gpuValuesBufferIn, 0, initialValues);

            // 2. Values Buffer Out (pong)
            const gpuValuesBufferOut = device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, // Need COPY_DST for compute pass output
            });

            // 3. Staging Buffer (for readback)
            const gpuStagingBuffer = device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            // --- Create Bind Groups (for ping-pong) ---
            const bindGroupA = device.createBindGroup({
                layout: this.gpuBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.gpuParamsBuffer } },
                    { binding: 1, resource: { buffer: this.gpuNodeInfosBuffer } },
                    { binding: 2, resource: { buffer: this.gpuConnectionsBuffer } },
                    { binding: 3, resource: { buffer: gpuValuesBufferIn } },   // Read from In
                    { binding: 4, resource: { buffer: gpuValuesBufferOut } },  // Write to Out
                ]
            });

            const bindGroupB = device.createBindGroup({
                layout: this.gpuBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.gpuParamsBuffer } },
                    { binding: 1, resource: { buffer: this.gpuNodeInfosBuffer } },
                    { binding: 2, resource: { buffer: this.gpuConnectionsBuffer } },
                    { binding: 3, resource: { buffer: gpuValuesBufferOut } }, // Read from Out
                    { binding: 4, resource: { buffer: gpuValuesBufferIn } },  // Write to In
                ]
            });


            // --- Command Encoding ---
            const commandEncoder = device.createCommandEncoder();
            const workgroupSize = 64; // Must match shader's @workgroup_size
            const workgroupCount = Math.ceil(maxNodeId / workgroupSize);

            let dispatchError = null;
            try {
                device.pushErrorScope('validation');
                for (let i = 0; i < iterations; i++) {
                    const passEncoder = commandEncoder.beginComputePass();
                    passEncoder.setPipeline(this.gpuComputePipeline);
                    passEncoder.setBindGroup(0, (i % 2 === 0) ? bindGroupA : bindGroupB);
                    passEncoder.dispatchWorkgroups(workgroupCount);
                    passEncoder.end();
                }
                dispatchError = await device.popErrorScope(); // Check AFTER loop
            } catch (err) {
                 console.error("Error during dispatch block:", err);
                 // Cleanup potentially created buffers before throwing
                 gpuValuesBufferIn?.destroy();
                 gpuValuesBufferOut?.destroy();
                 gpuStagingBuffer?.destroy();
                 throw err; // Re-throw
            }
        
            if (dispatchError) {
                console.error("Dispatch Error Scope:", dispatchError);
                 // Decide how to handle - maybe destroy buffers and throw?
                 gpuValuesBufferIn?.destroy();
                 gpuValuesBufferOut?.destroy();
                 gpuStagingBuffer?.destroy();
                throw new Error(`GPU Dispatch failed: ${dispatchError.message}`);
            }

            const finalResultBuffer = (iterations % 2 === 1) ? gpuValuesBufferOut : gpuValuesBufferIn; // Corrected Logic

            commandEncoder.copyBufferToBuffer(
                finalResultBuffer, 0, // Source buffer containing the final result
                gpuStagingBuffer, 0,  // Destination staging buffer
                bufferSize            // Size to copy
            );
    
            // --- Submit and Readback ---
            device.queue.submit([commandEncoder.finish()]);
    
            // Wait for the GPU to finish and map the staging buffer
            await gpuStagingBuffer.mapAsync(GPUMapMode.READ, 0, bufferSize);
            const copyArrayBuffer = gpuStagingBuffer.getMappedRange(0, bufferSize);
            // IMPORTANT: Create a new Float32Array or copy the data (.slice(0))
            // before unmapping, as the underlying ArrayBuffer becomes detached.
            const finalValuesData = new Float32Array(copyArrayBuffer.slice(0));
            gpuStagingBuffer.unmap();
    
            // Optional: Add logging here to see the first few values right after readback
            // console.log("Raw Readback Data (first 10):", finalValuesData.slice(0, 10));
    
            // Log specific hidden values (adjust indices if needed)
            // Find IDs of interest first if they aren't static
            const hiddenNodeIdsToLog = [27, 28, 29]; // Example IDs
            const loggedVals = hiddenNodeIdsToLog.map(id => (id < finalValuesData.length) ? finalValuesData[id]?.toFixed(4) : 'N/A');
            console.log(`Final Hidden Vals Readback (IDs ${hiddenNodeIdsToLog.join(', ')}): ${loggedVals.join(', ')}`);
    
    
            // --- Cleanup Per-Invocation Buffers ---
            gpuValuesBufferIn.destroy();
            gpuValuesBufferOut.destroy();
            gpuStagingBuffer.destroy();
    
    
            // --- Extract Output Node Values ---
            const outputNodes = [...this.nodes.values()].filter(n => n.type === 'output');
            const outputResult = outputNodes.map(node => {
                // Safety check for node ID vs buffer size
                if (node.id < maxNodeId && node.id < finalValuesData.length) {
                     return finalValuesData[node.id];
                } else {
                    console.warn(`Output node ID ${node.id} is out of bounds for readback buffer (size ${finalValuesData.length}). Returning 0.`);
                    return 0;
                }
            });
    
            // Optional: Log extracted output values
            console.log("Extracted Output Values:", outputResult);
    
            return outputResult;
        }

        // Optional: Method to clean up persistent GPU resources
        destroyGPU() {
            this.gpuNodeInfosBuffer?.destroy();
            this.gpuConnectionsBuffer?.destroy();
            this.gpuParamsBuffer?.destroy();
            // Note: Pipeline/Layouts don't have explicit destroy methods usually.
            // Destroying the device invalidates everything.
            // this.gpuDevice?.destroy(); // Careful: destroys the device for ALL networks using it

            this.gpuDevice = null;
            this.gpuComputePipeline = null;
            this.gpuNodeInfosBuffer = null;
            this.gpuConnectionsBuffer = null;
            this.gpuParamsBuffer = null;
            this.gpuBindGroupLayout = null;
            console.log("Persistent GPU resources destroyed (except device).");
        }
    }

    // ------------------------------
    // Example usage:
    //
    // // Create a NEAT network with 3 inputs and 2 outputs
    // const network = new NeuralNetwork(3, 2);
    //
    // // Run a feedforward pass with sample input values
    // const output = network.feedForward([0.5, -0.2, 0.8]);
    // console.log("Network output:", output);
    //
    // // Mutate weights, add connection, and add node mutations
    // network.mutateWeights();
    // network.mutateAddConnection();
    // network.mutateAddNode();
    //
    // // Create a second network (e.g., a mutated copy) and perform crossover
    // const network2 = new NeuralNetwork(3, 2);
    // const childNetwork = NeuralNetwork.crossover(network, network2);
    // console.log("Child network output:", childNetwork.feedForward([0.5, -0.2, 0.8]));
    // ------------------------------
    //
    // You can integrate this class into your reinforcement learning system,
    // evolving a population of such networks with selection, speciation, and fitness evaluation.


    // Car Class
    class Car {
        constructor(x, y, angle = 0) {
            // Position and dimensions
            this.x = x;
            this.y = y;
            this.width = 20;
            this.height = 10;
            this.angle = angle;

            // Physics properties
            this.speed = 0;
            this.maxSpeed = 5;
            this.acceleration = 0.1;
            this.deceleration = 0.05;
            this.turnSpeed = 0.05;

            // State
            this.crashed = false;
            this.finishReached = false;
            this.distanceTraveled = 0;

            // Controls
            this.controls = {
                forward: false,
                backward: false,
                left: false,
                right: false
            };

            // Lidar data
            this.lidarData = Array(LIDAR_COUNT).fill(LIDAR_RANGE);

            // Is this car controllable by user
            this.isControllable = false;
        }

        update(trackPoints, startPoint, tempCtx) {
            if (this.finishReached) return;

            // Update speed based on input
            if (this.controls.forward) {
                this.speed += this.acceleration;
            } else if (this.controls.backward) {
                this.speed -= this.acceleration;
            } else {
                // Apply friction
                if (this.speed > 0) {
                    this.speed -= this.deceleration;
                    if (this.speed < 0) this.speed = 0;
                } else if (this.speed < 0) {
                    this.speed += this.deceleration;
                    if (this.speed > 0) this.speed = 0;
                }
            }

            // Cap speed
            this.speed = Math.max(Math.min(this.speed, this.maxSpeed), -this.maxSpeed / 2);

            // Update angle based on input and speed
            if (this.speed !== 0) {
                const turnFactor = Math.abs(this.speed) / this.maxSpeed; // Turn faster at higher speeds
                if (this.controls.left) {
                    this.angle -= this.turnSpeed * turnFactor;
                }
                if (this.controls.right) {
                    this.angle += this.turnSpeed * turnFactor;
                }
            }

            // Replace the entire distance calculation and position update code:

            // Calculate the previous position
            const previousX = this.x;
            const previousY = this.y;

            // Update position based on angle and speed (ONLY ONCE)
            this.x += Math.cos(this.angle) * this.speed;
            this.y += Math.sin(this.angle) * this.speed;

            if (trackPoints.length > 1) {
                // Determine finish point based on track type
                let finishPointX, finishPointY;
                if (startPoint && Math.hypot(trackPoints[trackPoints.length - 1].x - startPoint.x,
                    trackPoints[trackPoints.length - 1].y - startPoint.y) < pointConnectionThreshold) {
                    // For circular tracks, use start point as finish
                    finishPointX = startPoint.x;
                    finishPointY = startPoint.y;
                } else {
                    // For open tracks, use last point as finish
                    finishPointX = trackPoints[trackPoints.length - 1].x;
                    finishPointY = trackPoints[trackPoints.length - 1].y;
                }

                // Find the nearest point on the track path to the current position
                // Replace the track progress calculation part in the car's update method:

                // Find the nearest point on the track path to the current position
                let minDist = Infinity;
                let nearestPointIdx = 0;

                for (let i = 0; i < trackPoints.length; i++) {
                    const dist = Math.hypot(this.x - trackPoints[i].x, this.y - trackPoints[i].y);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestPointIdx = i;
                    }
                }

                // Only count progress if we're moving forward and not crashed
                if (this.speed > 0 && !this.crashed) {
                    // Improved track progress calculation with interpolation
                    // Get the nearest point and the next point on the track
                    const currentPoint = trackPoints[nearestPointIdx];
                    const nextPointIdx = Math.min(nearestPointIdx + 1, trackPoints.length - 1);
                    const nextPoint = trackPoints[nextPointIdx];

                    // Calculate the fractional position between the current and next point
                    let interpolationFactor = 0;
                    if (nearestPointIdx < trackPoints.length - 1) {
                        const segmentVector = {
                            x: nextPoint.x - currentPoint.x,
                            y: nextPoint.y - currentPoint.y
                        };
                        const segmentLength = Math.hypot(segmentVector.x, segmentVector.y);

                        if (segmentLength > 0) {
                            // Project car position onto the segment between nearest and next point
                            const carVector = {
                                x: this.x - currentPoint.x,
                                y: this.y - currentPoint.y
                            };

                            // Calculate dot product to find projection
                            const dotProduct = (carVector.x * segmentVector.x + carVector.y * segmentVector.y) / segmentLength;
                            interpolationFactor = Math.max(0, Math.min(1, dotProduct / segmentLength));
                        }
                    }

                    // Calculate more precise track progress with interpolation
                    const preciseProgress = (nearestPointIdx + interpolationFactor) / (trackPoints.length - 1);
                    this.trackProgress = preciseProgress;

                    // Calculate distance based on how far along the track we are
                    const estimatedTrackLength = calculateTrackLength(trackPoints);
                    this.distanceTraveled = preciseProgress * estimatedTrackLength;
                }
            } else {
                // Fallback for incomplete tracks
                if (this.speed > 0 && !this.crashed) {
                    const movementX = this.x - previousX;
                    const movementY = this.y - previousY;
                    const distanceMoved = Math.sqrt(movementX * movementX + movementY * movementY);
                    this.distanceTraveled += distanceMoved;
                }
            }

            // The checkCollision and other functions will follow...

            // Check for collision with track borders
            this.checkCollision(tempCtx);

            // Check if car reached finish point
            this.checkFinish(trackPoints, startPoint, pointConnectionThreshold);

            // Update lidar readings
            this.updateLidar(tempCtx);
        }

        draw(ctx) {
            // Draw lidar rays first (so they appear behind the car)
            //this.drawLidar(ctx);

            ctx.save();
            ctx.translate(this.x, this.y);
            ctx.rotate(this.angle);

            // Car body
            ctx.fillStyle = this.crashed ? '#e74c3c' : '#3498db';
            ctx.fillRect(-this.width / 2, -this.height / 2, this.width, this.height);

            // Car front (to show direction)
            ctx.fillStyle = '#2c3e50';
            ctx.fillRect(this.width / 2 - 5, -this.height / 2, 5, this.height);

            ctx.restore();
        }

        drawLidar(ctx) {
            ctx.lineWidth = 1;

            for (let i = 0; i < LIDAR_COUNT; i++) {
                const angle = this.angle + (i * (2 * Math.PI / LIDAR_COUNT));
                const rayLength = this.lidarData[i];

                const endX = this.x + Math.cos(angle) * rayLength;
                const endY = this.y + Math.sin(angle) * rayLength;

                // Determine if the ray is hitting an obstacle (outside track)
                const isHittingObstacle = rayLength < LIDAR_RANGE;

                ctx.strokeStyle = isHittingObstacle ? LIDAR_COLOR_DANGER : LIDAR_COLOR_SAFE;
                ctx.beginPath();
                ctx.moveTo(this.x, this.y);
                ctx.lineTo(endX, endY);
                ctx.stroke();
            }
        }

        updateLidar(tempCtx) {
            // Check each lidar ray for collisions
            for (let i = 0; i < LIDAR_COUNT; i++) {
                const angle = this.angle + (i * (2 * Math.PI / LIDAR_COUNT));
                let rayLength = LIDAR_RANGE;

                // Ray casting algorithm to find collision
                for (let j = 1; j <= LIDAR_RANGE; j++) {
                    const checkX = this.x + Math.cos(angle) * j;
                    const checkY = this.y + Math.sin(angle) * j;

                    // Check if this point is outside canvas bounds
                    if (checkX < 0 || checkX >= tempCtx.canvas.width ||
                        checkY < 0 || checkY >= tempCtx.canvas.height) {
                        rayLength = j;
                        break;
                    }

                    // Check if this point is outside the track
                    try {
                        const pixel = tempCtx.getImageData(checkX, checkY, 1, 1).data;
                        if (pixel[3] === 0) { // If transparent (outside track)
                            rayLength = j;
                            break;
                        }
                    } catch (e) {
                        // If error (outside canvas), consider it a hit
                        rayLength = j;
                        break;
                    }
                }

                this.lidarData[i] = rayLength;
            }
        }

        // Now improve the collision detection by adding more check points
        // Replace the checkCollision method with this improved version:

        checkCollision(tempCtx) {
            // Create more check points to better detect collisions
            const checkPoints = [
                // Center
                { x: this.x, y: this.y },

                // Front, back, left, right (same as before)
                { x: this.x + Math.cos(this.angle) * (this.width / 2), y: this.y + Math.sin(this.angle) * (this.width / 2) }, // Front
                { x: this.x - Math.cos(this.angle) * (this.width / 2), y: this.y - Math.sin(this.angle) * (this.width / 2) }, // Back
                { x: this.x + Math.sin(this.angle) * (this.height / 2), y: this.y - Math.cos(this.angle) * (this.height / 2) }, // Right
                { x: this.x - Math.sin(this.angle) * (this.height / 2), y: this.y + Math.cos(this.angle) * (this.height / 2) }, // Left

                // Add corners for better detection
                // Front-right corner
                {
                    x: this.x + Math.cos(this.angle) * (this.width / 2) + Math.sin(this.angle) * (this.height / 2),
                    y: this.y + Math.sin(this.angle) * (this.width / 2) - Math.cos(this.angle) * (this.height / 2)
                },
                // Front-left corner
                {
                    x: this.x + Math.cos(this.angle) * (this.width / 2) - Math.sin(this.angle) * (this.height / 2),
                    y: this.y + Math.sin(this.angle) * (this.width / 2) + Math.cos(this.angle) * (this.height / 2)
                },
                // Back-right corner
                {
                    x: this.x - Math.cos(this.angle) * (this.width / 2) + Math.sin(this.angle) * (this.height / 2),
                    y: this.y - Math.sin(this.angle) * (this.width / 2) - Math.cos(this.angle) * (this.height / 2)
                },
                // Back-left corner
                {
                    x: this.x - Math.cos(this.angle) * (this.width / 2) - Math.sin(this.angle) * (this.height / 2),
                    y: this.y - Math.sin(this.angle) * (this.width / 2) + Math.cos(this.angle) * (this.height / 2)
                }
            ];

            // Check if any point is on the track
            let onTrack = false;
            for (const point of checkPoints) {
                try {
                    const pixel = tempCtx.getImageData(point.x, point.y, 1, 1).data;
                    // Check if pixel is track color (not transparent)
                    if (pixel[3] > 0) {
                        onTrack = true;
                        break;
                    }
                } catch (e) {
                    // If point is outside canvas, consider it off-track
                    continue;
                }
            }

            // Update car state
            if (!onTrack && !this.crashed) {
                this.crashed = true;
                this.speed *= 0.3; // Slow down when crashed
                if (this.isControllable) {
                    canvasContainer.classList.add('collision');
                    updateStatus('Crashed! Stay on the track.');
                }
            } else if (onTrack && this.crashed) {
                this.crashed = false;
                if (this.isControllable) {
                    canvasContainer.classList.remove('collision');
                    updateStatus('Testing track... Use arrow keys or WASD to drive');
                }
            }
        }

        checkFinish(trackPoints, startPoint, threshold) {
            // If already reached finish or no track points, return
            if (this.finishReached || !trackPoints.length) return;

            const lastPoint = trackPoints[trackPoints.length - 1];
            const isCircuit = startPoint &&
                Math.hypot(lastPoint.x - startPoint.x, lastPoint.y - startPoint.y) < threshold;

            let finishPoint;

            if (isCircuit) {
                // For circuit tracks, finish is at start point
                finishPoint = startPoint;
            } else {
                // For open tracks, finish is at end point
                finishPoint = lastPoint;
            }

            // Check if car is at finish point
            const distanceToFinish = Math.hypot(this.x - finishPoint.x, this.y - finishPoint.y);
            if (distanceToFinish < 20) { // Within 20px of finish point
                this.finishReached = true;
                if (this.isControllable) {
                    showFinishNotification();
                    updateStatus('Finish reached! Distance: ' + Math.round(this.distanceTraveled) + 'px');
                }
            }
        }
    }

    // RLAgent class: combines a car and its NEAT neural network controller.
    class RLAgent {
        constructor(startPoint, network = null) {
            // Create a new car at the start point.
            this.car = new Car(startPoint.x, startPoint.y);
            // If no network is provided, create a new one with inputs for lidar + speed + angle
            // Input size remains LIDAR_COUNT + 2
            this.network = network || new NeuralNetwork(LIDAR_COUNT + 2, 4);
            this.fitness = 0;
            this.lastProgress = 0; // Track progress to encourage forward movement
            this.stagnationCounter = 0; // Counter for how long the agent hasn't progressed
            this.maxStagnation = 150; // Number of frames allowed without progress before penalty/early termination (optional)
        }

        // Update the agent for one frame.
        update(trackPoints, startPoint, tempCtx) {
            // --- Network calculation and control setting is now done in runSimulationTick ---
            // REMOVE OR COMMENT OUT:
            // const normalizedLidarData = this.car.lidarData.map(distance => distance / LIDAR_RANGE);
            // const inputs = [ /* ... */ ];
            // const outputs = this.network.feedForward(inputs); // <-- REMOVE THIS LINE
            // this.car.controls.left = outputs[0] > 0.5;       // <-- REMOVE THIS LINE
            // this.car.controls.right = outputs[1] > 0.5;      // <-- REMOVE THIS LINE
            // this.car.controls.forward = outputs[2] > 0.5;    // <-- REMOVE THIS LINE
            // this.car.controls.backward = outputs[3] > 0.5 && !this.car.controls.forward; // <-- REMOVE THIS LINE

            // --- Keep the car physics update ---
            // Update the car's physics and sensors based on controls set elsewhere (in runSimulationTick)
            this.car.update(trackPoints, startPoint, tempCtx);

            // --- Fitness calculation is now done in runSimulationTick after car.update ---
            // REMOVE OR COMMENT OUT:
            // this.fitness = this.car.distanceTraveled;
            // if (this.car.finishReached) { /* ... */ }
            // if (this.car.distanceTraveled > this.lastProgress) { /* ... */ } else { /* ... */ }
            // if (this.stagnationCounter > this.maxStagnation) { /* ... */ }
        }

        // Draw the agent's car.
        // This method is defined later using RLAgent.prototype.draw = function (ctx) { ... }
        // No changes needed here for normalization.
    }
    // Add this to the RLAgent class's draw method
    // Add this right after the class RLAgent constructor or at the end of the class definition
    RLAgent.prototype.draw = function (ctx) {
        const isLeader = controlledCar === this.car;

        // Store the original car color
        const originalCarColor = this.car.crashed ? '#e74c3c' : '#3498db';

        // If this is the leading car, make it more visible
        if (isLeader) {
            this.car.width = 24;  // Make it slightly larger
            ctx.shadowColor = '#ffff00';
            ctx.shadowBlur = 10;
        } else {
            this.car.width = 20;  // Normal size
        }

        // Draw the car
        this.car.draw(ctx);

        // Reset shadow effects
        ctx.shadowBlur = 0;
    };

    let rlAgents = [];
    const populationSize = 10;
    let simulationTimer = null;
    const simulationDuration = 45000; // e.g., 15 seconds per generation

    // Add this function to help calculate the track length
    function calculateTrackLength(points) {
        let length = 0;
        for (let i = 1; i < points.length; i++) {
            length += Math.hypot(
                points[i].x - points[i - 1].x,
                points[i].y - points[i - 1].y
            );
        }
        return length;
    }

    // Start the RL simulation (evolve while the "Run RL" mode is active)
    // Replace your startRLSimulation function with this updated version
    function startRLSimulation() {
        // Check if track exists
        if (points.length < 2 || !startPoint) {
            updateStatus('Draw a track first!');
            return;
        }

        // Create a new population of agents
        rlAgents = [];
        crashedAgentsHistory = []; // Reset crashed agents history
        generationCount = 1; // Reset generation counter

        for (let i = 0; i < populationSize; i++) {
            rlAgents.push(new RLAgent(startPoint));
        }

        // Show car canvas and hide instructions like in startTesting
        isTesting = true;
        runRLBtn.classList.add('active');
        carCanvas.classList.remove('hidden');
        distanceCounter.classList.remove('hidden');
        carInstructions.classList.add('hidden'); // Hide keyboard controls
        trackCanvas.style.cursor = 'default';
        hideFinishNotification();

        // Activate camera and set initial position
        camera.active = true;
        camera.targetScale = 1.5; // Zoom in a bit
        camera.targetX = trackCanvas.width / 2 - startPoint.x * camera.targetScale;
        camera.targetY = trackCanvas.height / 2 - startPoint.y * camera.targetScale;

        // Start simulation loop
        simulationTimer = setTimeout(endGeneration, simulationDuration);
        updateStatus('RL Simulation: Generation 1 started');

        // Clear any existing animation before starting new one
        if (animationId) cancelAnimationFrame(animationId);
        gameLoopRL();
    }

    // Add this function to stop the RL simulation
    function stopRLSimulation() {
        isTesting = false;
        clearTimeout(simulationTimer);
        runRLBtn.classList.remove('active');

        // Reset and clear everything
        rlAgents = [];
        carCanvas.classList.add('hidden');
        distanceCounter.classList.add('hidden');
        trackCanvas.style.cursor = 'crosshair';

        // Deactivate camera
        resetCamera();
        controlledCar = null;

        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Redraw track
        if (points.length > 0) {
            drawTrack();
        }
    }

    // Modify the runRLBtn click handler to toggle between start and stop
    runRLBtn.addEventListener('click', () => {
        if (isTesting) {
            stopRLSimulation();
        } else {
            startRLSimulation();
        }
    });


    stopRLBtn.addEventListener('click', () => {
        endGeneration();
    });



    // Add this function to provide better status information during RL training
    // Update the updateRLStatus function to show more info
    function updateRLStatus() {
        if (!isTesting || (rlAgents.length === 0 && crashedAgentsHistory.length === 0)) return;

        // Find farthest active agent
        let farthestAgent = rlAgents.length > 0 ?
            rlAgents.reduce((max, agent) => agent.car.distanceTraveled > max.car.distanceTraveled ? agent : max, rlAgents[0]) :
            null;

        // Count agents
        const survivors = rlAgents.length;
        const crashed = crashedAgentsHistory.length;

        // Calculate best distance - consider both active and crashed agents
        let bestDistance = 0;
        if (farthestAgent) {
            bestDistance = farthestAgent.car.distanceTraveled;
        }

        // Check if any crashed agent went further
        for (let crashedAgent of crashedAgentsHistory) {
            if (crashedAgent.car.distanceTraveled > bestDistance) {
                bestDistance = crashedAgent.car.distanceTraveled;
            }
        }

        // Update status and distance counter
        updateStatus(`Generation ${generationCount}: ${survivors} alive, ${crashed} crashed`);
        updateDistanceCounter(bestDistance);

        // If any agent reached the finish, highlight this
        const anyFinished = [...rlAgents, ...crashedAgentsHistory].some(agent => agent.car.finishReached);
        if (anyFinished) {
            showFinishNotification();
        }
    }

    // Call this function at the end of each gameLoopRL iteration
    // Add this line at the end of gameLoopRL function before requestAnimationFrame:
    updateRLStatus();

    // End the generation: evaluate fitness, select best agents, and produce the next generation.
    // Replace your endGeneration function with this updated version
    // Inside endGeneration function...

    function endGeneration() {
        clearTimeout(simulationTimer);
        if (!isTesting) return; // Exit if testing stopped

        console.log("Generation completed. Evolving population...");
        let allAgents = [...rlAgents, ...crashedAgentsHistory]; // Combine for evaluation
        allAgents.sort((a, b) => b.fitness - a.fitness);

        // ... (rest of the evaluation, logging, best network visualization) ...
        if (allAgents.length === 0) {
            console.warn("No agents available for evolution. Restarting population.");
            // Handle this case - maybe create a default population?
            rlAgents = [];
            for (let i = 0; i < populationSize; i++) {
                rlAgents.push(new RLAgent(startPoint));
            }
            crashedAgentsHistory = [];
            generationCount++;
            // Need to init GPU for the new population
            if (rlAgents.length > 0) {
                rlAgents[0].network.initGPU().then(() => { // Ensure GPU ready async
                    rlAgents[0].network.updateGPUStructureBuffers(); // Update buffers for new default pop
                    updateStatus(`Generation ${generationCount}: Restarted population.`);
                    simulationTimer = setTimeout(endGeneration, simulationDuration);
                    if (!animationId) gameLoopRL(); // Restart loop if stopped
                });
            }
            return;
        }

        visualizeNetwork(allAgents[0].network); // Visualize best

        // Evolve: Generate new population
        let newPopulation = [];
        let bestAgent = allAgents[0];
        newPopulation.push(new RLAgent(startPoint, bestAgent.network)); // Keep best

        while (newPopulation.length < populationSize) {
            // ... (crossover and mutation logic - remains the same) ...
            let childNetwork;
            if (allAgents.length > 1) {
                let parent1 = allAgents[Math.floor(Math.random() * Math.min(3, allAgents.length))];
                let parent2 = allAgents[Math.floor(Math.random() * Math.min(5, allAgents.length))];
                childNetwork = NeuralNetwork.crossover(parent1.network, parent2.network);
            } else {
                childNetwork = NeuralNetwork.crossover(bestAgent.network, bestAgent.network);
            }
            childNetwork.mutateWeights();
// Inside the while loop in endGeneration:
            const triggerActivationMutationChance = 0.3; // Chance to *attempt* activation mutation for the network
            const perNodeActivationMutationChance = 0.1; // Chance *per node* to mutate if triggered

            if (Math.random() < triggerActivationMutationChance) {
                childNetwork.mutateActivations(perNodeActivationMutationChance);
            }
            if (Math.random() < 0.2) childNetwork.mutateAddNode();
            if (Math.random() < 0.3) childNetwork.mutateActivations();
            newPopulation.push(new RLAgent(startPoint, childNetwork));
        }

        // --- Update GPU Buffers for the NEW population ---
        // Ensure GPU is initialized (might be first generation or after restart)
        const gpuUpdatePromise = newPopulation[0].network.initGPU().then(() => {
            // Call updateGPUStructureBuffers ONCE using one network from the new population
            // This assumes all networks in the new generation are structurally representable
            // by the maxNodeId and connection count derived from the NeuralNetwork class state.
            console.log("Updating GPU buffers for new generation...");
            newPopulation[0].network.updateGPUStructureBuffers();
        }).catch(err => {
            console.error("Failed to initialize/update GPU for new generation:", err);
            // Handle error - maybe fall back to CPU?
            stopRLSimulation(); // Stop if GPU setup fails
            updateStatus("GPU Error during generation update. Stopping.");
        });

        // --- Reset State AFTER potentially async GPU update ---
        gpuUpdatePromise.then(() => {
            // Continue only if still testing after async GPU update
            if (!isTesting) return;

            rlAgents = newPopulation; // Assign the new population
            crashedAgentsHistory = []; // Clear history
            generationCount++;

            // Reset cars to start position
            for (const agent of rlAgents) {
                agent.car.x = startPoint.x;
                agent.car.y = startPoint.y;
                agent.car.angle = 0; // Reset angle
                agent.car.speed = 0;
                agent.car.crashed = false;
                agent.car.finishReached = false;
                agent.car.distanceTraveled = 0;
                agent.lastProgress = 0; // Reset progress tracking
                agent.stagnationCounter = 0;
            }

            updateStatus(`Generation ${generationCount}: ${rlAgents.length} alive, 0 crashed`);

            // Restart simulation timer for the next generation
            simulationTimer = setTimeout(endGeneration, simulationDuration);

            // Ensure the game loop continues if it wasn't already running
            if (!animationId && isTesting) {
                console.log("Restarting game loop for new generation.");
                gameLoopRL();
            }
        }); // End of .then block for GPU update
    }

    // Also, add this function to ensure animation frames continue properly:
    function ensureRLAnimationContinues() {
        // Check if RL mode is active but no animation is running
        if (isTesting && !animationId && rlAgents.length > 0) {
            console.log("Restarting animation frame that stopped unexpectedly");
            gameLoopRL();
        }
    }

    // RL game loop: update and draw all RL agents.
    // Replace your gameLoopRL function with this improved version
    // Replace the camera logic in your gameLoopRL function with this version
    // Modify the gameLoopRL function to remove crashed agents from visualization
    // ... (keep existing variables like animationId, isTesting, etc.) ...

    let isRunningTick = false; // Flag to prevent overlapping async ticks

    // The main animation loop function (remains synchronous for requestAnimationFrame)
    function gameLoopRL() {
        // If not testing OR if a tick is already in progress, just schedule the next frame and exit
        if (!isTesting || isRunningTick) {
            // Ensure animation continues if testing is true
            if (isTesting) {
                animationId = requestAnimationFrame(gameLoopRL);
            } else {
                animationId = null; // Ensure it stops if testing becomes false
            }
            return;
        }

        isRunningTick = true; // Set flag: simulation tick starting

        // Run the asynchronous simulation logic
        runSimulationTick()
            .then(() => {
                // Simulation tick finished successfully
                isRunningTick = false; // Clear flag
                // Schedule the next frame IF still testing
                if (isTesting) {
                    animationId = requestAnimationFrame(gameLoopRL);
                } else {
                    animationId = null; // Ensure it stops if testing became false during async work
                }
            })
            .catch(error => {
                // Handle errors during the simulation tick
                console.error("Error during simulation tick:", error);
                isRunningTick = false; // Clear flag
                stopRLSimulation(); // Stop the simulation on critical error
                updateStatus(`Simulation Error: ${error.message}`);
            });
    }

    // The new asynchronous function containing the core simulation logic for one frame/tick
    async function runSimulationTick() {
        // --- Check if stopped ---
        if (!isTesting || rlAgents.length === 0) { // Check if any agents left
            // If no agents, we might need to trigger endGeneration logic cleanly
            if (isTesting && rlAgents.length === 0 && crashedAgentsHistory.length > 0) {
                console.log("All agents crashed/finished, ending generation from tick check.");
                endGeneration(); // Trigger end generation logic
            }
            return; // Stop this tick if not testing or no agents
        }

        // 1. Prepare inputs for all active agents
        const inputsArray = rlAgents.map(agent => {
            const normalizedLidarData = agent.car.lidarData.map(d => d / LIDAR_RANGE);
            // Ensure inputs are floats
            return [
                ...normalizedLidarData.map(Number),
                Number(agent.car.speed / agent.car.maxSpeed),
                Number(agent.car.angle / (2 * Math.PI)) // Normalize angle, can wrap around but ok
            ];
        });

        // 2. Initiate all GPU feedforward calls
        // Ensure GPU is initialized (should ideally happen once before starting)
        if (rlAgents.length > 0 && !rlAgents[0].network.gpuDevice) {
            console.log("Initializing GPU for the first time...");
            await rlAgents[0].network.initGPU();
            // Assuming updateGPUStructureBuffers was called after population creation/mutation
        }

        const gpuPromises = rlAgents.map((agent, index) =>
            agent.network.feedForwardGPU(inputsArray[index], 1000)
                .catch(err => { // Add error handling per agent
                    console.error(`GPU feedforward failed for agent ${index} (Node ${agent.network.nodes.size}, Conn ${agent.network.connections.length}):`, err);
                    // Set crashed state or provide default output if GPU fails
                    agent.car.crashed = true;
                    return [0, 0, 0, 0]; // Default output (e.g., no action)
                })
        );

        // 3. Wait for all GPU results
        const allOutputs = await Promise.all(gpuPromises);

        // --- Check if stopped while waiting ---
        if (!isTesting) return;

        // 4. Apply outputs to car controls for active agents
        rlAgents.forEach((agent, index) => {
            // Check if agent crashed during GPU step (due to error)
            if (agent.car.crashed) return;

            const outputs = allOutputs[index];
            // Basic Thresholding - adjust thresholds (0.5) if needed
            agent.car.controls.left = outputs[0] > 0.5;
            agent.car.controls.right = outputs[1] > 0.5;
            agent.car.controls.forward = outputs[2] > 0.5;
            // Prevent moving forward and backward simultaneously
            agent.car.controls.backward = outputs[3] > 0.5 && !agent.car.controls.forward;

            if (rlAgents.length > 0 && index < 2) { // Log for first 2 agents
                console.log(`Agent ${index} RAW Outputs:`, allOutputs[index].map(v => v.toFixed(4))); // Format numbers
            }
        });

        // 5. Update car physics, collision, lidar (synchronous part)
        const tempCtx = createCollisionCanvas(); // This function uses the global 'points' internally
        for (const agent of rlAgents) {
            // Update car physics and sensors based on controls set above
            // ***** FIX HERE: Use 'points' and 'startPoint' *****
            agent.car.update(points, startPoint, tempCtx); // <-- Pass the correct global variables

            // Recalculate fitness and stagnation based on new car state
            agent.fitness = agent.car.distanceTraveled;
            if (agent.car.finishReached) {
                agent.fitness += 10000; // Bonus for finishing
            }


            if (agent.car.distanceTraveled > agent.lastProgress + 0.1) { // Add small threshold
                agent.lastProgress = agent.car.distanceTraveled;
                agent.stagnationCounter = 0;
            } else if (!agent.car.finishReached && !agent.car.crashed) {
                agent.stagnationCounter++;
            }

            if (agent.stagnationCounter > agent.maxStagnation) {
                agent.fitness -= 50; // Penalty for stagnation
                // Optionally force crash: agent.car.crashed = true;
            }
        }

        // 6. Handle crashed agents
        const newlyCrashedAgents = rlAgents.filter(agent => agent.car.crashed);
        if (newlyCrashedAgents.length > 0) {
            crashedAgentsHistory.push(...newlyCrashedAgents); // Add to history
            rlAgents = rlAgents.filter(agent => !agent.car.crashed); // Remove from active list
        }

        // 7. Check if simulation should end early (all crashed/finished)
        if (rlAgents.length === 0 && crashedAgentsHistory.length > 0) { // Check history length > 0
            console.log("All agents finished or crashed! Ending generation early.");
            endGeneration(); // Call original endGeneration
            return; // Stop this tick
        }
        // Handle the case where all agents might finish without crashing
        const allFinished = rlAgents.every(agent => agent.car.finishReached);
        if (allFinished && rlAgents.length > 0) {
            console.log("All active agents finished! Ending generation early.");
            crashedAgentsHistory.push(...rlAgents); // Move finishers to history for evaluation
            rlAgents = []; // Clear active agents
            endGeneration();
            return; // Stop this tick
        }


        // 8. Update camera target (find farthest *active* agent)
        if (rlAgents.length > 0) {
            let farthestAgent = rlAgents.reduce((farthest, current) =>
                current.car.distanceTraveled > farthest.car.distanceTraveled ? current : farthest,
                rlAgents[0]
            );
            makeControllableVehicle(farthestAgent.car); // Update leader display
            // Camera target is updated within updateCamera based on controlledCar
        } else {
            controlledCar = null; // No active cars left to control/follow
        }

        // 9. Update camera position smoothly
        updateCamera();

        // 10. Draw track and cars (only active agents)
        drawTrack(); // Draws track based on camera
        carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height); // Clear car canvas

        if (camera.active) carCtx.save(); // Apply camera if active
        if (camera.active) applyCameraTransform(carCtx);

        for (const agent of rlAgents) { // Draw only active agents
            agent.draw(carCtx);
        }

        if (camera.active) carCtx.restore(); // Restore context if camera was applied


        // 11. Update UI Status
        updateRLStatus();

    } // End of runSimulationTick



    // Make a car controllable by the user
    function makeControllableVehicle(car) {
        // Check if there's already a controllable car
        if (controlledCar) {
            // Remove control from previous car
            controlledCar.isControllable = false;
        }

        // Set this car as the controlled car
        car.isControllable = true;
        controlledCar = car;

        // Update UI to show car's data
        updateDistanceCounter(car.distanceTraveled);
    }

    // Set canvas dimensions
    function resizeCanvas() {
        trackCanvas.width = canvasContainer.offsetWidth;
        trackCanvas.height = canvasContainer.offsetHeight;
        carCanvas.width = trackCanvas.width;
        carCanvas.height = trackCanvas.height;
    }

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Camera system
    const camera = {
        x: 0,
        y: 0,
        scale: 1,
        targetX: 0,
        targetY: 0,
        targetScale: 1,
        lerpFactor: 0.05,
        active: false
    };

    // Track drawing state
    let isDrawing = false;
    let points = [];
    let startPoint = null;

    // Game stat
    let controlledCar = null;
    let cars = [];

    // Game loop variables
    let animationId = null;

    // Initialize
    function init() {
        // Stop simulations first
        if (isTesting && runRLBtn.classList.contains('active')) {
            stopRLSimulation();
        } else if (isTesting) {
            stopTesting();
        }

        // Reset track data
        points = [];
        startPoint = null;
        isDrawing = false;

        // Reset car/agent data
        cars = [];
        rlAgents = [];
        crashedAgentsHistory = [];
        controlledCar = null;

        // Reset UI elements
        updateStatus('Draw a track...');
        clearCanvas(); // Clears both canvases
        canvasContainer.classList.remove('circuit-track', 'finish-track', 'collision');
        carInstructions.classList.add('hidden');
        distanceCounter.classList.add('hidden');
        hideFinishNotification();
        networkVisualization.classList.add('hidden'); // Hide network viz on clear

        // Reset camera
        resetCamera();

        // Cancel any animation frame
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        // Clear simulation timer if any
        if (simulationTimer) {
            clearTimeout(simulationTimer);
            simulationTimer = null;
        }

        console.log("Track and simulation state cleared.");
    }

    // Clear canvas
    function clearCanvas() {
        trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height);
        carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height);
        canvasContainer.classList.remove('circuit-track', 'finish-track', 'collision');
    }

    // Update status text
    function updateStatus(text) {
        statusText.textContent = text;
    }

    // Draw the track
    function drawTrack() {
        clearCanvas();

        if (points.length === 0) return;

        if (camera.active) {
            // Apply camera transformation when active
            trackCtx.save();
            applyCameraTransform(trackCtx);
        }

        // Draw track path
        trackCtx.beginPath();
        trackCtx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
            trackCtx.lineTo(points[i].x, points[i].y);
        }

        trackCtx.strokeStyle = trackColor;
        trackCtx.lineWidth = trackWidth;
        trackCtx.lineCap = 'round';
        trackCtx.lineJoin = 'round';
        trackCtx.stroke();

        // Draw start point
        if (startPoint) {
            trackCtx.beginPath();
            trackCtx.arc(startPoint.x, startPoint.y, startPointRadius, 0, Math.PI * 2);
            trackCtx.fillStyle = startPointColor;
            trackCtx.fill();
        }

        // Draw end point if not a circuit
        const lastPoint = points[points.length - 1];
        const isCircuit = startPoint &&
            Math.hypot(lastPoint.x - startPoint.x, lastPoint.y - startPoint.y) < pointConnectionThreshold;

        if (points.length > 1 && !isCircuit) {
            trackCtx.beginPath();
            trackCtx.arc(lastPoint.x, lastPoint.y, startPointRadius, 0, Math.PI * 2);
            trackCtx.fillStyle = finishPointColor;
            trackCtx.fill();
        }

        if (camera.active) {
            trackCtx.restore();
        }

        // Update status
        if (points.length > 1) {
            if (isCircuit) {
                updateStatus('Circuit Track Completed');
                canvasContainer.classList.add('circuit-track');
            } else {
                updateStatus('Open Track with Finish Point');
                canvasContainer.classList.add('finish-track');
            }
        }
    }

    // Event listeners for track drawing
    trackCanvas.addEventListener('mousedown', (e) => {
        if (isTesting) return;

        isDrawing = true;
        const rect = trackCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // If this is the first point, set it as start point
        if (points.length === 0) {
            startPoint = { x, y };
        }

        points.push({ x, y });
        drawTrack();
    });

    trackCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawing || isTesting) return;

        const rect = trackCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        points.push({ x, y });
        drawTrack();
    });

    trackCanvas.addEventListener('mouseup', () => {
        if (isTesting) return;

        isDrawing = false;
        checkTrackCompletion();
    });

    trackCanvas.addEventListener('mouseleave', () => {
        if (isDrawing && !isTesting) {
            isDrawing = false;
            checkTrackCompletion();
        }
    });

    // Check if track is a circuit or has a finish point
    function checkTrackCompletion() {
        if (points.length <= 1 || !startPoint) return;

        const lastPoint = points[points.length - 1];
        const distance = Math.hypot(lastPoint.x - startPoint.x, lastPoint.y - startPoint.y);

        if (distance < pointConnectionThreshold) {
            // Complete the circuit by adding the start point
            points.push({ x: startPoint.x, y: startPoint.y });
            drawTrack();
        }
    }

    // Camera functions
    function resetCamera() {
        camera.x = 0;
        camera.y = 0;
        camera.scale = 1;
        camera.targetX = 0;
        camera.targetY = 0;
        camera.targetScale = 1;
        camera.active = false;
    }

    function updateCamera() {
        if (!camera.active || !controlledCar) return;

        // Lerp camera position towards target (smooth following)
        camera.x += (camera.targetX - camera.x) * camera.lerpFactor;
        camera.y += (camera.targetY - camera.y) * camera.lerpFactor;
        camera.scale += (camera.targetScale - camera.scale) * camera.lerpFactor;

        // Update target to follow controlled car
        camera.targetX = trackCanvas.width / 2 - controlledCar.x * camera.targetScale;
        camera.targetY = trackCanvas.height / 2 - controlledCar.y * camera.targetScale;
    }

    function applyCameraTransform(ctx) {
        // Apply camera transformation (translate and scale)
        ctx.translate(camera.x, camera.y);
        ctx.translate(trackCanvas.width / 2, trackCanvas.height / 2);
        ctx.scale(camera.scale, camera.scale);
        ctx.translate(-trackCanvas.width / 2, -trackCanvas.height / 2);
    }

    // Car testing functions
    function startTesting() {
        if (points.length < 2 || !startPoint) {
            updateStatus('Draw a track first!');
            return;
        }

        isTesting = true;
        testBtn.classList.add('active');
        carCanvas.classList.remove('hidden');
        carInstructions.classList.remove('hidden');
        distanceCounter.classList.remove('hidden');
        trackCanvas.style.cursor = 'default';

        // Reset cars array
        cars = [];

        // Create a new car at the start point
        const newCar = new Car(startPoint.x, startPoint.y);
        cars.push(newCar);

        // Make the car controllable by the user
        makeControllableVehicle(newCar);

        // Activate camera and set initial position
        camera.active = true;
        camera.targetScale = 1.5; // Zoom in a bit
        camera.targetX = trackCanvas.width / 2 - newCar.x * camera.targetScale;
        camera.targetY = trackCanvas.height / 2 - newCar.y * camera.targetScale;

        updateStatus('Testing track... Use arrow keys or WASD to drive');
        updateDistanceCounter(0);
        hideFinishNotification();

        // Start game loop
        if (animationId) cancelAnimationFrame(animationId);
        gameLoop();
    }

    function stopTesting() {
        isTesting = false;
        testBtn.classList.remove('active');
        carCanvas.classList.add('hidden');
        carInstructions.classList.add('hidden');
        distanceCounter.classList.add('hidden');
        trackCanvas.style.cursor = 'crosshair';
        canvasContainer.classList.remove('collision');
        hideFinishNotification();

        // Deactivate camera and clear car references
        resetCamera();
        controlledCar = null;
        cars = [];

        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        // Redraw track
        if (points.length > 0) {
            drawTrack();
        }
    }

    // Update distance counter display
    function updateDistanceCounter(distance) {
        distanceCounter.textContent = 'Distance: ' + Math.round(distance) + 'px';
    }

    // Show finish notification
    function showFinishNotification() {
        finishNotification.classList.remove('hidden');
    }

    // Hide finish notification
    function hideFinishNotification() {
        finishNotification.classList.add('hidden');
    }

    // Create collision detection canvas
    function createCollisionCanvas() {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = trackCanvas.width;
        tempCanvas.height = trackCanvas.height;

        // Draw track for collision detection
        tempCtx.beginPath();
        tempCtx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
            tempCtx.lineTo(points[i].x, points[i].y);
        }

        tempCtx.strokeStyle = trackColor;
        tempCtx.lineWidth = trackWidth;
        tempCtx.lineCap = 'round';
        tempCtx.lineJoin = 'round';
        tempCtx.stroke();

        return tempCtx;
    }

    // Game loop
    function gameLoop() {
        // Create collision detection canvas
        const tempCtx = createCollisionCanvas();

        // Update all cars
        for (const car of cars) {
            car.update(points, startPoint, tempCtx);
        }

        // Update distance display for controlled car
        if (controlledCar) {
            updateDistanceCounter(controlledCar.distanceTraveled);
        }

        // Update camera
        updateCamera();

        // Draw everything
        drawTrack();

        // Draw cars on car canvas
        carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height);

        if (camera.active) {
            carCtx.save();
            applyCameraTransform(carCtx);
        }

        // Draw all cars
        for (const car of cars) {
            car.draw(carCtx);
        }

        if (camera.active) {
            carCtx.restore();
        }

        // Continue animation
        animationId = requestAnimationFrame(gameLoop);
    }

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        if (!isTesting || !controlledCar) return;

        switch (e.key) {
            case 'ArrowUp':
            case 'w':
                controlledCar.controls.forward = true;
                break;
            case 'ArrowDown':
            case 's':
                controlledCar.controls.backward = true;
                break;
            case 'ArrowLeft':
            case 'a':
                controlledCar.controls.left = true;
                break;
            case 'ArrowRight':
            case 'd':
                controlledCar.controls.right = true;
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        if (!isTesting || !controlledCar) return;

        switch (e.key) {
            case 'ArrowUp':
            case 'w':
                controlledCar.controls.forward = false;
                break;
            case 'ArrowDown':
            case 's':
                controlledCar.controls.backward = false;
                break;
            case 'ArrowLeft':
            case 'a':
                controlledCar.controls.left = false;
                break;
            case 'ArrowRight':
            case 'd':
                controlledCar.controls.right = false;
                break;
        }
    });

    // Button listeners
    clearBtn.addEventListener('click', init);

    testBtn.addEventListener('click', () => {
        if (isTesting) {
            stopTesting();
        } else {
            startTesting();
        }
    });

    exportTrackBtn.addEventListener('click', exportTrackData);
    importTrackBtn.addEventListener('click', importTrackData);

    // Make sure the initial call to init happens at the end
    init();



    // Visualize the neural network
    function visualizeNetwork(network) {
        if (!network) return;

        // Store the best network for visualization
        bestNetwork = network;

        // Show the visualization panel
        networkVisualization.classList.remove('hidden');

        // Set up canvas dimensions
        networkCanvas.width = networkCanvas.offsetWidth;
        networkCanvas.height = networkCanvas.offsetHeight;

        // Clear canvas
        networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

        // Calculate visualization parameters
        const padding = 30;
        const width = networkCanvas.width - padding * 2;
        const height = networkCanvas.height - padding * 2;

        // Group nodes by type
        const inputNodes = Array.from(network.nodes.values()).filter(node => node.type === 'input');
        const hiddenNodes = Array.from(network.nodes.values()).filter(node => node.type === 'hidden');
        const outputNodes = Array.from(network.nodes.values()).filter(node => node.type === 'output');
        const biasNodes = Array.from(network.nodes.values()).filter(node => node.type === 'bias');

        // Calculate layer positions
        const layers = [];
        layers.push([...inputNodes, ...biasNodes]); // Input layer includes bias nodes

        // If we have hidden nodes, add a hidden layer
        if (hiddenNodes.length > 0) {
            layers.push(hiddenNodes);
        }

        layers.push(outputNodes); // Output layer

        // Calculate node positions
        const nodePositions = new Map();

        for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            const layer = layers[layerIndex];
            const x = padding + (width * layerIndex) / (layers.length - 1);

            for (let nodeIndex = 0; nodeIndex < layer.length; nodeIndex++) {
                const node = layer[nodeIndex];
                const y = padding + (height * nodeIndex) / Math.max(layer.length - 1, 1);
                nodePositions.set(node.id, { x, y });
            }
        }

        // Draw connections first (so they appear behind nodes)
        for (const conn of network.connections) {
            if (!conn.enabled) continue; // Skip disabled connections

            const fromPos = nodePositions.get(conn.inNode);
            const toPos = nodePositions.get(conn.outNode);

            if (!fromPos || !toPos) continue;

            // Draw connection
            networkCtx.beginPath();
            networkCtx.moveTo(fromPos.x, fromPos.y);
            networkCtx.lineTo(toPos.x, toPos.y);

            // Set connection color based on weight value
            const alpha = Math.min(1.0, Math.abs(conn.weight) * 0.5 + 0.2);
            const colorStr = conn.weight > 0 ?
                `rgba(0, 128, 0, ${alpha})` :
                `rgba(255, 0, 0, ${alpha})`;

            networkCtx.strokeStyle = colorStr;
            networkCtx.lineWidth = Math.min(5, Math.abs(conn.weight) * 2 + 1);
            networkCtx.stroke();
        }

        // Draw nodes
        const nodeRadius = 8;
        for (const [nodeId, pos] of nodePositions.entries()) {
            const node = network.nodes.get(nodeId);

            networkCtx.beginPath();
            networkCtx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);

            // Set node color based on type
            switch (node.type) {
                case 'input':
                    networkCtx.fillStyle = '#3498db'; // Blue
                    break;
                case 'hidden':
                    networkCtx.fillStyle = '#f39c12'; // Orange
                    break;
                case 'output':
                    networkCtx.fillStyle = '#2ecc71'; // Green
                    break;
                case 'bias':
                    networkCtx.fillStyle = '#9b59b6'; // Purple
                    break;
                default:
                    networkCtx.fillStyle = '#95a5a6'; // Gray
            }

            networkCtx.fill();

            // Add a label if it's an activation function other than sigmoid
            if (node.activationType && node.activationType !== 'sigmoid') {
                networkCtx.fillStyle = '#333';
                networkCtx.font = '8px Arial';
                networkCtx.textAlign = 'center';
                networkCtx.fillText(node.activationType.charAt(0).toUpperCase(), pos.x, pos.y + 3);
            }
        }

        // Update network stats
        updateNetworkStats(network);
    }

    // Update network statistics display
    function updateNetworkStats(network) {
        if (!network) return;

        // Count nodes by type
        const inputCount = Array.from(network.nodes.values()).filter(node => node.type === 'input').length;
        const hiddenCount = Array.from(network.nodes.values()).filter(node => node.type === 'hidden').length;
        const outputCount = Array.from(network.nodes.values()).filter(node => node.type === 'output').length;
        const biasCount = Array.from(network.nodes.values()).filter(node => node.type === 'bias').length;

        // Count connections
        const enabledConnections = network.connections.filter(conn => conn.enabled).length;
        const disabledConnections = network.connections.length - enabledConnections;

        // Count activation functions
        const activationsMap = new Map();
        for (const node of network.nodes.values()) {
            if (!activationsMap.has(node.activationType)) {
                activationsMap.set(node.activationType, 0);
            }
            activationsMap.set(node.activationType, activationsMap.get(node.activationType) + 1);
        }

        // Format activation stats
        let activationText = "Activations: ";
        for (const [type, count] of activationsMap.entries()) {
            activationText += `${type}: ${count}, `;
        }
        activationText = activationText.slice(0, -2); // Remove trailing comma

        // Update stats elements
        activationStats.textContent = activationText;
        nodesStats.textContent = `Nodes: Input: ${inputCount}, Hidden: ${hiddenCount}, Output: ${outputCount}, Bias: ${biasCount}`;
        connectionsStats.textContent = `Connections: Active: ${enabledConnections}, Disabled: ${disabledConnections}`;
    }

    // Function to serialize a neural network to JSON
    function serializeNetwork(network) {
        if (!network) return null;

        // Create a serializable object from the network
        const serialized = {
            nodes: [],
            connections: [],
            nextNodeId: network.nextNodeId,
            nextInnovationNumber: network.nextInnovationNumber
        };

        // Serialize nodes
        for (const [id, node] of network.nodes.entries()) {
            serialized.nodes.push({
                id: node.id,
                type: node.type,
                activationType: node.activationType || 'sigmoid'
            });
        }

        // Serialize connections
        for (const conn of network.connections) {
            serialized.connections.push({
                inNode: conn.inNode,
                outNode: conn.outNode,
                weight: conn.weight,
                enabled: conn.enabled,
                innovation: conn.innovation
            });
        }

        return serialized;
    }

    function exportTrackData() {
        if (points.length < 2 || !startPoint) {
            updateStatus('No track data to export. Draw a track first.');
            return;
        }

        // Prepare the data object
        const trackData = {
            timestamp: new Date().toISOString(),
            startPoint: startPoint,
            points: points
        };

        // Convert to JSON string
        const jsonString = JSON.stringify(trackData, null, 2); // Pretty print JSON

        // Create a blob and download link
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        // Create temporary link and trigger download
        const a = document.createElement('a');
        a.href = url;
        // Suggest a filename
        a.download = `track_data_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();

        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            updateStatus(`Track successfully exported as ${a.download}`);
        }, 0);
    }

    // --- NEW: Track Import Function ---
    function importTrackData() {
        // Stop any ongoing simulation before importing
        if (isTesting && runRLBtn.classList.contains('active')) {
            stopRLSimulation(); // Stop RL if running
        } else if (isTesting) {
            stopTesting(); // Stop manual testing if running
        }

        // Trigger the hidden file input
        trackFileInput.click();
    }

    // --- NEW: Handle Track File Input ---
    trackFileInput.addEventListener('change', (event) => {
        if (event.target.files.length === 0) {
            return; // No file selected
        }

        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const trackData = JSON.parse(e.target.result);

                // Basic validation
                if (!trackData || !trackData.startPoint || !Array.isArray(trackData.points)) {
                    throw new Error("Invalid track file format. Missing 'startPoint' or 'points' array.");
                }
                if (trackData.points.length < 2) {
                    throw new Error("Track data must contain at least two points.");
                }

                // --- Load the track data ---
                // Clear existing track first (important!)
                init(); // Use init to reset everything cleanly

                // Assign loaded data
                startPoint = trackData.startPoint;
                points = trackData.points;

                // Redraw the newly loaded track
                drawTrack();

                updateStatus(`Track successfully imported from ${file.name}.`);

            } catch (error) {
                console.error("Error importing track:", error);
                updateStatus(`Error importing track: ${error.message}`);
                // Optionally clear the track if import fails badly
                // init();
                // drawTrack();
            } finally {
                // Clear the file input value so the user can select the same file again
                event.target.value = '';
            }
        };

        reader.onerror = () => {
            console.error("Error reading track file");
            updateStatus('Error reading track file.');
            event.target.value = ''; // Clear the input
        };

        reader.readAsText(file);
    });

    // Export the best model to JSON file
    function exportBestModel() {
        if (!bestNetwork) {
            updateStatus('No best network available to export');
            return;
        }

        // Serialize the network
        const serialized = serializeNetwork(bestNetwork);

        // Add metadata
        const modelData = {
            timestamp: new Date().toISOString(),
            generation: generationCount,
            networkStructure: {
                inputSize: Array.from(bestNetwork.nodes.values()).filter(n => n.type === 'input').length,
                outputSize: Array.from(bestNetwork.nodes.values()).filter(n => n.type === 'output').length,
                hiddenNodes: Array.from(bestNetwork.nodes.values()).filter(n => n.type === 'hidden').length,
            },
            network: serialized
        };

        // Convert to JSON string
        const jsonString = JSON.stringify(modelData, null, 2);

        // Create a blob and download link
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        // Create temporary link and trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = `nn_model_gen${generationCount}_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();

        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            // Show success animation
            exportModelBtn.classList.add('success');
            setTimeout(() => {
                exportModelBtn.classList.remove('success');
            }, 2000);

            updateStatus(`Model successfully exported as ${a.download}`);
        }, 0);
    }

    // Function to deserialize a network from JSON (for future use)
    function deserializeNetwork(serialized) {
        if (!serialized) return null;

        const network = new NeuralNetwork(0, 0); // Create empty network

        // Clear default nodes and connections
        network.nodes = new Map();
        network.connections = [];

        // Restore nextNodeId and nextInnovationNumber
        network.nextNodeId = serialized.nextNodeId;
        network.nextInnovationNumber = serialized.nextInnovationNumber;

        // Restore nodes
        for (const nodeData of serialized.nodes) {
            const node = new NodeGene(
                nodeData.id,
                nodeData.type,
                nodeData.activationType || 'sigmoid'
            );
            network.nodes.set(node.id, node);
        }

        // Restore connections
        for (const connData of serialized.connections) {
            const conn = new ConnectionGene(
                connData.inNode,
                connData.outNode,
                connData.weight,
                connData.enabled,
                connData.innovation
            );
            network.connections.push(conn);
        }

        return network;
    }

    // Add event listener for export button
    exportModelBtn.addEventListener('click', () => {
        exportBestModel();
    });

    // Import model from JSON file
    function importModel() {
        // Trigger the hidden file input
        modelFileInput.click();
    }

    // Handle model file selection
    modelFileInput.addEventListener('change', (event) => {
        if (event.target.files.length === 0) {
            return;
        }

        // Pause the current RL simulation if it's running
        const wasRunning = isTesting;
        if (wasRunning) {
            clearTimeout(simulationTimer);
        }

        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const modelData = JSON.parse(e.target.result);

                if (!modelData.network) {
                    throw new Error("Invalid model file format");
                }

                // Deserialize the network
                const importedNetwork = deserializeNetwork(modelData.network);

                if (!importedNetwork) {
                    throw new Error("Failed to import network");
                }

                // Store as best network for visualization
                bestNetwork = importedNetwork;

                // Visualize the imported network
                visualizeNetwork(importedNetwork);

                // If we were in RL mode, replace the current population with this network
                if (wasRunning) {
                    // Create a new population using the imported network
                    rlAgents = [];
                    crashedAgentsHistory = [];

                    // Create one agent with exact copy of imported network
                    rlAgents.push(new RLAgent(startPoint, importedNetwork));

                    // Create the rest with mutations of the imported network
                    for (let i = 1; i < populationSize; i++) {
                        // Clone the network and apply some mutations
                        const clonedNetwork = NeuralNetwork.crossover(importedNetwork, importedNetwork);
                        clonedNetwork.mutateWeights(0.5); // Lower mutation rate for imported models

                        if (Math.random() < 0.2) clonedNetwork.mutateAddConnection();
                        if (Math.random() < 0.1) clonedNetwork.mutateAddNode();

                        rlAgents.push(new RLAgent(startPoint, clonedNetwork));
                    }

                    // Show success message and animation
                    importModelBtn.classList.add('success');
                    setTimeout(() => {
                        importModelBtn.classList.remove('success');
                    }, 2000);

                    // Resume the simulation
                    updateStatus(`Imported model from generation ${modelData.generation || 'unknown'}. Continuing with ${rlAgents.length} agents.`);

                    // Restart the simulation timer
                    simulationTimer = setTimeout(endGeneration, simulationDuration);

                    // Make sure visualization is updated
                    if (animationId) cancelAnimationFrame(animationId);
                    gameLoopRL();
                } else {
                    // Just display the network visualization
                    importModelBtn.classList.add('success');
                    setTimeout(() => {
                        importModelBtn.classList.remove('success');
                    }, 2000);
                    updateStatus(`Model imported from generation ${modelData.generation || 'unknown'}. Click Run RL to use it.`);
                }

            } catch (error) {
                console.error("Error importing model:", error);
                updateStatus(`Error importing model: ${error.message}`);
            }

            // Clear the file input for future imports
            modelFileInput.value = '';
        };

        reader.onerror = () => {
            console.error("Error reading file");
            updateStatus('Error reading file');
            modelFileInput.value = '';
        };

        reader.readAsText(file);
    });

    // Add event listener for import button
    importModelBtn.addEventListener('click', importModel);
});