document.addEventListener('DOMContentLoaded', () => {
    const undoBtn = document.getElementById('undo-btn'); // Get the new button
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
    let isPanning = false;
    let panStartX = 0;
    let panStartY = 0;
    let cameraStartX = 0;
    let cameraStartY = 0;

    const COMPATIBILITY_THRESHOLD = 3.0; // Adjust this based on observation
    const C1 = 1.0; // Weight for excess genes
    const C2 = 1.0; // Weight for disjoint genes
    const C3 = 0.5; // Weight for average weight difference of matching genes
    const MIN_SPECIES_SIZE_FOR_ELITE = 3; // Minimum members for a species champion to be auto-kept
    const STAGNATION_THRESHOLD = 15; // Generations a species can go without improvement before penalties/removal
    // --- End NEAT Speciation Parameters ---

    let currentSpecies = []; // Array to hold Species objects across generations
    let nextSpeciesId = 0;   // Simple ID counter for species

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
    const SIMULATION_DURATION = 45000;
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
    }

    class Species {
        constructor(representativeAgent, id) {
            this.id = id;
            this.representative = representativeAgent.network; // The network representing the species
            this.members = [representativeAgent];             // Agents belonging to this species
            this.totalAdjustedFitness = 0;
            this.offspringAllocation = 0;
            this.bestFitness = representativeAgent.fitness; // Track best fitness achieved by this species
            this.generationsSinceImprovement = 0;           // For stagnation tracking
        }

        addMember(agent) {
            this.members.push(agent);
        }

        // Calculate shared fitness for the species
        calculateAdjustedFitness() {
            if (this.members.length === 0) {
                this.totalAdjustedFitness = 0;
                return;
            }
            let totalRawFitness = 0;
            for (const member of this.members) {
                totalRawFitness += member.fitness;
            }
            // Fitness sharing: divide total raw fitness by species size
            this.totalAdjustedFitness = totalRawFitness / this.members.length;

            // Update stagnation counter
            const currentBestFitness = Math.max(0, ...this.members.map(m => m.fitness)); // Max fitness in this gen
            if (currentBestFitness > this.bestFitness) {
                this.bestFitness = currentBestFitness;
                this.generationsSinceImprovement = 0;
            } else {
                this.generationsSinceImprovement++;
            }
        }

        // Select a parent using tournament selection *within* the species
        selectParent(tournamentSize = 3) {
            if (this.members.length === 0) return null;
            if (this.members.length === 1) return this.members[0];

            let tournament = [];
            for (let i = 0; i < Math.min(tournamentSize, this.members.length); i++) {
                tournament.push(this.members[Math.floor(Math.random() * this.members.length)]);
            }
            // Return the member with the highest raw fitness in the tournament
            return tournament.reduce((best, current) => (current.fitness > best.fitness ? current : best), tournament[0]);
        }

        // Get the best performing member of this species in the current generation
        getBestMember() {
            if (this.members.length === 0) return null;
            return this.members.reduce((best, current) => (current.fitness > best.fitness ? current : best), this.members[0]);
        }


        // Prepare for the next generation
        reset(keepRepresentative = true) {
            // Optionally update representative (e.g., pick a random member)
            if (!keepRepresentative && this.members.length > 0) {
                this.representative = this.members[Math.floor(Math.random() * this.members.length)].network;
            }
            this.members = []; // Clear members for the next generation's assignment
            this.totalAdjustedFitness = 0;
            this.offspringAllocation = 0;
            // Stagnation and bestFitness carry over
        }
    }

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
            this.nodes = new Map();
            this.connections = [];
            this.nextNodeId = 0;
            this.nextInnovationNumber = 0;

            // Create input nodes.
            const inputNodes = [];
            for (let i = 0; i < inputSize; i++) {
                const node = new NodeGene(this.nextNodeId++, 'input');
                this.nodes.set(node.id, node);
                inputNodes.push(node);
            }

            // Create a bias node.
            const bias = new NodeGene(this.nextNodeId++, 'bias');
            this.nodes.set(bias.id, bias);
            const inputAndBiasNodes = [...inputNodes, bias]; // Combine for connections

            // Create output nodes.
            const outputNodes = [];
            for (let i = 0; i < outputSize; i++) {
                const node = new NodeGene(this.nextNodeId++, 'output'); // Use default activation initially, mutation can change it
                this.nodes.set(node.id, node);
                outputNodes.push(node);
            }

            // --- Minimal Initial Connections ---
            // Connect all input and bias nodes directly to all output nodes
            for (let inNode of inputAndBiasNodes) {
                for (let outNode of outputNodes) {
                    this.addConnection(inNode.id, outNode.id, Math.random() * 2 - 1);
                }
            }

            // Fully connect hidden nodes to output nodes
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

        // Add this static method inside the NeuralNetwork class
        static compatibilityDistance(network1, network2, c1, c2, c3) {
            const genes1 = new Map();
            network1.connections.forEach(conn => genes1.set(conn.innovation, conn));

            const genes2 = new Map();
            network2.connections.forEach(conn => genes2.set(conn.innovation, conn));

            let matchingGenes = 0;
            let disjointGenes = 0;
            let excessGenes = 0;
            let weightDifference = 0;

            const maxInnovation1 = network1.connections.length > 0 ?
                Math.max(...network1.connections.map(c => c.innovation)) : 0;
            const maxInnovation2 = network2.connections.length > 0 ?
                Math.max(...network2.connections.map(c => c.innovation)) : 0;
            const maxOverallInnovation = Math.max(maxInnovation1, maxInnovation2);

            // Iterate up to the highest innovation number found in either genome
            for (let i = 0; i <= maxOverallInnovation; i++) {
                const gene1 = genes1.get(i);
                const gene2 = genes2.get(i);

                if (gene1 && gene2) {
                    // Matching gene
                    matchingGenes++;
                    weightDifference += Math.abs(gene1.weight - gene2.weight);
                } else if (gene1 && !gene2) {
                    // Gene in 1, not in 2
                    if (i <= maxInnovation2) {
                        disjointGenes++; // Disjoint if within range of genome 2
                    } else {
                        excessGenes++; // Excess if beyond range of genome 2
                    }
                } else if (!gene1 && gene2) {
                    // Gene in 2, not in 1
                    if (i <= maxInnovation1) {
                        disjointGenes++; // Disjoint if within range of genome 1
                    } else {
                        excessGenes++; // Excess if beyond range of genome 1
                    }
                }
                // If neither exists, do nothing
            }

            // Avoid division by zero for weight difference
            if (matchingGenes > 0) {
                weightDifference /= matchingGenes;
            }

            // N is the number of genes in the larger genome (important normalization factor)
            // Handle cases with zero connections
            const N = Math.max(1, network1.connections.length, network2.connections.length);

            // The compatibility distance formula
            const distance = (c1 * excessGenes / N) + (c2 * disjointGenes / N) + (c3 * weightDifference);

            // console.log(`Dist: E=${excessGenes}, D=${disjointGenes}, W=${weightDifference.toFixed(2)}, N=${N} => ${distance.toFixed(2)}`); // DEBUG
            return distance;
        }

        // Computes the network output for a given array of input values.
        // It uses a simple iterative algorithm to ensure that each node is processed once its inputs are ready.
        feedForward(inputArray) {
            // Reset all node values.
            for (let node of this.nodes.values()) {
                node.value = 0;
            }

            // Set input node values.
            const inputNodes = [...this.nodes.values()].filter(n => n.type === 'input');
            if (inputArray.length !== inputNodes.length) {
                throw new Error("Input array length does not match number of input nodes.");
            }
            for (let i = 0; i < inputNodes.length; i++) {
                inputNodes[i].value = inputArray[i];
            }

            // Set bias node value.
            const biasNodes = [...this.nodes.values()].filter(n => n.type === 'bias');
            for (let b of biasNodes) {
                b.value = 1;
            }

            // Process remaining nodes.
            // We assume the network is acyclic; therefore, we process nodes only when all their incoming values are computed.
            const remaining = new Set(this.nodes.keys());
            // Remove input and bias nodes (their values are already set).
            for (let node of this.nodes.values()) {
                if (node.type === 'input' || node.type === 'bias') {
                    remaining.delete(node.id);
                }
            }

            let maxIterations = this.nodes.size;
            while (remaining.size > 0 && maxIterations-- > 0) {
                for (let nodeId of Array.from(remaining)) {
                    const node = this.nodes.get(nodeId);
                    // Find all enabled incoming connections.
                    const incoming = this.connections.filter(conn => conn.outNode === nodeId && conn.enabled);
                    // Check if all input nodes for these connections have been computed.
                    const ready = incoming.every(conn => !remaining.has(conn.inNode));
                    if (ready) {
                        // Sum weighted inputs.
                        let sum = 0;
                        for (let conn of incoming) {
                            const inNode = this.nodes.get(conn.inNode);
                            sum += inNode.value * conn.weight;
                        }
                        // Activate the node.
                        node.value = node.activate(sum);
                        remaining.delete(nodeId);
                    }
                }
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

        mutateToggleEnable() {
            // If there are no connections, we can't toggle anything
            if (this.connections.length === 0) {
                return;
            }

            // Select a random connection index
            const randomIndex = Math.floor(Math.random() * this.connections.length);
            const connection = this.connections[randomIndex];

            // Flip the enabled status
            connection.enabled = !connection.enabled;

            // Optional: Add logging to see when this happens
            // console.log(`Toggled connection ${connection.innovation} (${connection.inNode} -> ${connection.outNode}) to ${connection.enabled}`);
        }

        // Add this to the NeuralNetwork class
        mutateActivations(probability = 0.3) {
            const activationTypes = ['sigmoid', 'relu', 'leaky_relu', 'tanh', 'linear'];

            for (let [nodeId, node] of this.nodes.entries()) {
                // Only mutate hidden and output nodes (keep input and bias with default)
                if (node.type === 'hidden' || node.type === 'output') {
                    if (Math.random() < probability) {
                        // Choose a random activation function that's different from current
                        let newType;
                        do {
                            newType = activationTypes[Math.floor(Math.random() * activationTypes.length)];
                        } while (newType === node.activationType && activationTypes.length > 1);

                        node.activationType = newType;
                    }
                }
            }
        }

        // Performs crossover between two parent networks.
        // This method assumes both parents have the same input/output sizes.
        // Matching genes are randomly chosen from either parent while disjoint/excess genes are inherited from parent1.
        static crossover(parent1, parent2) {
            // Create a child network with no initial nodes (we'll copy nodes manually).
            const child = new NeuralNetwork(0, 0);
            child.nodes = new Map();
            child.connections = [];

            // Copy nodes from parent1.
            for (let [id, node] of parent1.nodes.entries()) {
                if (parent2.nodes.has(id) && (node.type === 'hidden' || node.type === 'output')) {
                    const activationType = Math.random() < 0.5 ?
                        node.activationType : parent2.nodes.get(id).activationType;
                    child.nodes.set(id, new NodeGene(node.id, node.type, activationType));
                } else {
                    child.nodes.set(id, new NodeGene(node.id, node.type, node.activationType));
                }
            }
            // *** Important: Ensure nextNodeId is correctly set early ***
            // Use the maximum from parents PLUS potentially new nodes from parent2 if a more complex node union was done.
            // Since we only copy from parent1 nodes, max of parents should suffice for now.
            child.nextNodeId = Math.max(parent1.nextNodeId, parent2.nextNodeId);
            child.nextInnovationNumber = Math.max(parent1.nextInnovationNumber, parent2.nextInnovationNumber);


            // Map parent2's connections by their innovation number.
            const parent2Conns = new Map();
            for (let conn of parent2.connections) {
                parent2Conns.set(conn.innovation, conn);
            }

            // Process each connection in parent1.
            for (let conn1 of parent1.connections) {
                let conn2 = parent2Conns.get(conn1.innovation);
                let chosenConn = null;
                let chosenEnabled = true; // Default

                if (conn2 !== undefined) {
                    // Matching gene exists in both parents
                    chosenConn = Math.random() < 0.5 ? conn1 : conn2;
                    // Handle enabled status: Often, if disabled in either parent, it's disabled in child
                    chosenEnabled = conn1.enabled && conn2.enabled;
                } else {
                    // Disjoint or excess gene – inherit from parent1.
                    chosenConn = conn1;
                    chosenEnabled = conn1.enabled;
                }

                // *** FIX: Check if both referenced nodes exist in the child's node map ***
                if (child.nodes.has(chosenConn.inNode) && child.nodes.has(chosenConn.outNode)) {
                    // Copy the chosen connection gene into the child.
                    child.connections.push(new ConnectionGene(
                        chosenConn.inNode,
                        chosenConn.outNode,
                        chosenConn.weight,
                        // Apply the determined enabled status (or simply chosenConn.enabled if preferred)
                        chosenEnabled,
                        chosenConn.innovation
                    ));
                } else {
                    // Optional: Log if a connection is skipped
                    // console.log(`Skipping connection ${chosenConn.innovation} during crossover: Node ${!child.nodes.has(chosenConn.inNode) ? chosenConn.inNode : chosenConn.outNode} not found in child.`);
                }
            }
            // Removed redundant setting of nextNodeId/nextInnovationNumber from here

            return child;
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
            this.maxSpeed = 3;
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
            this.network = network || new NeuralNetwork(LIDAR_COUNT + 2, 4);
            this.fitness = 0;
        }

        // Update the agent for one frame.
        // Update the agent for one frame.
        update(trackPoints, startPoint, tempCtx) {
            // --- Normalize Lidar Data ---
            // Map raw distances (0 to LIDAR_RANGE) to a normalized range [0, 1]
            // We want 1 to represent maximum danger (closest object) and 0 maximum range (clear).
            const normalizedLidar = this.car.lidarData.map(
                // distance / LIDAR_RANGE gives a value from 0 (close) to 1 (far)
                // 1 - (distance / LIDAR_RANGE) inverts this to 1 (close) to 0 (far)
                distance => 1 - (Math.min(distance, LIDAR_RANGE) / LIDAR_RANGE)
                // Added Math.min to ensure distance doesn't accidentally exceed LIDAR_RANGE
            );
        
            // Create new input array with NORMALIZED lidar data plus velocity and angle
            const inputs = [
                ...normalizedLidar,                 // Use the normalized lidar values
                this.car.speed / this.car.maxSpeed, // Normalized speed (already in range ~[-0.5, 1])
                this.car.angle / (2 * Math.PI)      // Normalized angle (can be outside [0,1], maybe wrap?)
                // Consider wrapping angle: (this.car.angle % (2 * Math.PI)) / (2 * Math.PI) if angle can grow indefinitely
            ];

            // Get the network outputs.
            const outputs = this.network.feedForward(inputs);

            // Interpret outputs:
            this.car.controls.left = outputs[0] > 0.5;
            this.car.controls.right = outputs[1] > 0.5;
            this.car.controls.forward = outputs[2] > 0.5;
            this.car.controls.backward = outputs[3] > 0.5;

            // Update the car.
            this.car.update(trackPoints, startPoint, tempCtx);

            // Define fitness
            this.fitness = this.car.distanceTraveled;
            if (this.car.finishReached) {
                this.fitness += 10000;
            }
        }

        // Draw the agent's car.
        draw(ctx) {
            this.car.draw(ctx);
        }
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
    const populationSize = 20;
    let simulationTimer = null;
    const simulationDuration = 45000; // e.g., 15 seconds per generation

    // Mutation Probabilities
    const WEIGHT_MUTATION_PROB = 0.8;     // Probability a connection's weight is mutated
    const WEIGHT_PERTURB_CHANCE = 0.9;    // If mutating, chance to perturb vs. replace (0.0-1.0)
    const ADD_CONNECTION_PROB = 0.3;     // Probability of adding a new connection
    const ADD_NODE_PROB = 0.2;         // Probability of adding a new node
    const MUTATE_ACTIVATION_PROB = 0.1;  // Probability a node's activation fn changes
    const TOGGLE_ENABLE_PROB = 0;      // Probability an existing connection is enabled/disabled

    // Mutation Parameters
    const WEIGHT_MUTATION_STEP = 0.1;   // Max amount to perturb weight by

    // Selection Parameters (Example - adjust as needed)
    const ELITISM_COUNT = 1;             // Keep the top N agents directly
    const TOURNAMENT_SIZE = 3;           // Size of tournament for selecting parents

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
    // Replace your endGeneration function with this updated version
    function endGeneration() {
        clearTimeout(simulationTimer);

        // Skip evolution if we're no longer in testing mode
        if (!isTesting) return;

        console.log(`--- Generation ${generationCount} Ending ---`);

        // Combine active and crashed agents for evaluation
        let allAgents = [...rlAgents, ...crashedAgentsHistory];

        if (allAgents.length === 0) {
            console.warn("No agents survived or were created. Cannot proceed with evolution.");
            // Optionally restart or stop simulation
            stopRLSimulation(); // Or try restarting with a new initial population
            return;
        }

        // --- 1. Speciation ---
        console.log("Speciating population...");
        // Reset members of existing species, keeping representatives
        currentSpecies.forEach(species => species.reset(true));
        let newSpeciesCreated = 0;

        for (const agent of allAgents) {
            let foundSpecies = false;
            for (const species of currentSpecies) {
                const dist = NeuralNetwork.compatibilityDistance(
                    agent.network,
                    species.representative,
                    C1, C2, C3
                );
                if (dist < COMPATIBILITY_THRESHOLD) {
                    species.addMember(agent);
                    foundSpecies = true;
                    break; // Add to the first matching species
                }
            }

            if (!foundSpecies) {
                // Create a new species for this agent
                const newSpecies = new Species(agent, nextSpeciesId++);
                currentSpecies.push(newSpecies);
                newSpeciesCreated++;
            }
        }

        // Remove species that ended up with no members (representatives might not match anyone)
        currentSpecies = currentSpecies.filter(species => species.members.length > 0);
        console.log(`Speciation complete: ${currentSpecies.length} species found (${newSpeciesCreated} new).`);
        currentSpecies.forEach(s => console.log(`  Species ${s.id}: ${s.members.length} members`));


        // --- 2. Calculate Adjusted Fitness (Fitness Sharing) & Handle Stagnation ---
        console.log("Calculating adjusted fitness and checking stagnation...");
        let globalTotalAdjustedFitness = 0;
        const survivingSpecies = []; // Keep track of species that aren't stagnant

        for (const species of currentSpecies) {
            species.calculateAdjustedFitness();

            if (species.generationsSinceImprovement > STAGNATION_THRESHOLD && currentSpecies.length > 1) { // Don't kill the last species
                console.log(`Species ${species.id} stagnant for ${species.generationsSinceImprovement} generations. Removing.`);
                // This species will not get offspring allocated
            } else {
                survivingSpecies.push(species);
                globalTotalAdjustedFitness += species.totalAdjustedFitness;
                console.log(`  Species ${species.id}: BestRaw=${species.bestFitness.toFixed(2)}, AdjustedFit=${species.totalAdjustedFitness.toFixed(2)}, Stagnation=${species.generationsSinceImprovement}`);
            }
        }

        // Update currentSpecies to only include survivors for reproduction
        currentSpecies = survivingSpecies;

        if (currentSpecies.length === 0 || globalTotalAdjustedFitness <= 0) {
            console.error("All species died or stagnated, or zero total fitness! Evolution cannot continue.");
            // Handle this - maybe restart with a new random population?
            stopRLSimulation();
            return;
        }
        console.log(`Total adjusted fitness across ${currentSpecies.length} surviving species: ${globalTotalAdjustedFitness.toFixed(2)}`);


        // --- 3. Offspring Allocation ---
        console.log("Allocating offspring...");
        let totalOffspringAllocated = 0;
        for (const species of currentSpecies) {
            // Allocate offspring proportionally to the species' adjusted fitness
            species.offspringAllocation = Math.floor(
                (species.totalAdjustedFitness / globalTotalAdjustedFitness) * populationSize
            );
            totalOffspringAllocated += species.offspringAllocation;
            console.log(`  Species ${species.id}: Allocating ${species.offspringAllocation} offspring.`);
        }

        // Distribute remaining slots (due to flooring) - give to best species
        let remainingOffspring = populationSize - totalOffspringAllocated;
        if (remainingOffspring > 0 && currentSpecies.length > 0) {
            // Sort species by best raw fitness achieved (desc) to give remainder slots
            currentSpecies.sort((a, b) => b.bestFitness - a.bestFitness);
            for (let i = 0; i < remainingOffspring; i++) {
                currentSpecies[i % currentSpecies.length].offspringAllocation++;
                console.log(`  Giving +1 remainder offspring to Species ${currentSpecies[i % currentSpecies.length].id}`);
            }
        }


        // --- 4. Reproduction (Elitism, Crossover, Mutation) ---
        console.log("Reproducing population...");
        let newPopulation = [];

        // Elitism: Keep the best individual from each species (if large enough), cloning its network
        for (const species of currentSpecies) {
            if (species.offspringAllocation > 0 && species.members.length >= MIN_SPECIES_SIZE_FOR_ELITE) {
                const champion = species.getBestMember();
                if (champion) {
                    console.log(`  Keeping champion from Species ${species.id} (Fitness: ${champion.fitness.toFixed(2)})`);
                    const eliteNetwork = NeuralNetwork.crossover(champion.network, champion.network); // Clone
                    newPopulation.push(new RLAgent(startPoint, eliteNetwork));
                    // Reduce allocation for this species by 1 as we added an elite
                    species.offspringAllocation--;
                }
            }
        }
        // Ensure population size isn't exceeded by elites alone if MIN_SPECIES_SIZE_FOR_ELITE is low
        while (newPopulation.length > populationSize) {
            newPopulation.pop(); // Remove excess elites if necessary (unlikely with reasonable settings)
        }

        // Crossover and Mutation for remaining slots
        for (const species of currentSpecies) {
            // console.log(`Reproducing for Species ${species.id}, need ${species.offspringAllocation} more.`); // DEBUG
            for (let i = 0; i < species.offspringAllocation; i++) {
                if (newPopulation.length >= populationSize) break; // Stop if population is full

                let childNetwork;
                let parent1 = species.selectParent();
                let parent2 = species.selectParent(); // Allow self-reproduction within species

                if (!parent1 || !parent2) {
                    console.warn(`Could not select parents for species ${species.id}. Skipping offspring.`);
                    continue;
                }

                // Crossover
                childNetwork = NeuralNetwork.crossover(parent1.network, parent2.network);

                // Apply Mutations
                childNetwork.mutateWeights(WEIGHT_MUTATION_PROB, WEIGHT_PERTURB_CHANCE, WEIGHT_MUTATION_STEP);
                if (Math.random() < ADD_CONNECTION_PROB) childNetwork.mutateAddConnection();
                if (Math.random() < ADD_NODE_PROB) childNetwork.mutateAddNode();
                if (Math.random() < MUTATE_ACTIVATION_PROB) childNetwork.mutateActivations();
                if (Math.random() < TOGGLE_ENABLE_PROB) childNetwork.mutateToggleEnable();

                newPopulation.push(new RLAgent(startPoint, childNetwork));
            }
            if (newPopulation.length >= populationSize) break; // Stop outer loop if full
        }

        // Fill any remaining spots (due to errors/edge cases) with clones/mutations of the best overall agent
        if (newPopulation.length < populationSize && allAgents.length > 0) {
            console.warn(`Population size (${newPopulation.length}) less than target (${populationSize}). Filling remaining spots.`);
            allAgents.sort((a, b) => b.fitness - a.fitness); // Find global best
            const globalBestNetwork = allAgents[0].network;
            while (newPopulation.length < populationSize) {
                const clone = NeuralNetwork.crossover(globalBestNetwork, globalBestNetwork); // Clone
                // Apply some mutation
                clone.mutateWeights(0.5, 0.9, 0.1);
                if (Math.random() < 0.1) clone.mutateAddConnection();
                if (Math.random() < 0.05) clone.mutateAddNode();
                newPopulation.push(new RLAgent(startPoint, clone));
            }
        }


        // --- 5. Prepare for Next Generation ---
        rlAgents = newPopulation; // The new generation becomes the active agents
        crashedAgentsHistory = []; // Clear crashed history
        generationCount++;

        // Reset cars to start position and clear temporary state
        for (const agent of rlAgents) {
            agent.car.x = startPoint.x;
            agent.car.y = startPoint.y;
            agent.car.angle = 0;
            agent.car.speed = 0;
            agent.car.crashed = false;
            agent.car.finishReached = false;
            agent.car.distanceTraveled = 0;
            agent.fitness = 0;
        }

        // Visualize the best network from the *previous* generation before starting new one
        allAgents.sort((a, b) => b.fitness - a.fitness); // Sort previous gen agents
        if (allAgents.length > 0) {
            visualizeNetwork(allAgents[0].network);
            console.log(`Generation ${generationCount - 1} Best Raw Fitness: ${allAgents[0].fitness.toFixed(2)}`);
        }

        // Start next generation simulation
        updateStatus(`Generation ${generationCount}: Starting with ${rlAgents.length} agents across ${currentSpecies.length} species.`);
        if (animationId) cancelAnimationFrame(animationId);
        gameLoopRL(); // Render the initial state of the new generation
        simulationTimer = setTimeout(endGeneration, SIMULATION_DURATION);
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
    function gameLoopRL() {
        if (!isTesting || isPanning) { // Pause RL loop if panning (important!)
            animationId = requestAnimationFrame(gameLoopRL); // Keep requesting frame
            return;
        };

        const tempCtx = createCollisionCanvas();

        // --- Update Agents (existing code) ---
        for (let i = rlAgents.length - 1; i >= 0; i--) {
            const agent = rlAgents[i];
            agent.update(points, startPoint, tempCtx);

            // Check crash and move to history
            if (agent.car.crashed) {
                if (!crashedAgentsHistory.some(histAgent => histAgent === agent)) { // Avoid duplicates if logic allows multiple checks
                    crashedAgentsHistory.push(agent);
                }
                rlAgents.splice(i, 1); // Remove from active agents
            }
        }


        // --- Check if simulation should end early (existing code) ---
        if (rlAgents.length === 0) {
            console.log("All agents crashed! Ending generation early.");
            endGeneration(); // This function now handles the clearTimeout
            return; // Stop this frame
        }

        // --- Find Farthest Agent and Update Camera Target (existing code) ---
        if (rlAgents.length > 0) {
            let farthestAgent = rlAgents.reduce((farthest, current) =>
                current.car.distanceTraveled > farthest.car.distanceTraveled ? current : farthest,
                rlAgents[0]
            );
            makeControllableVehicle(farthestAgent.car); // Update which car camera follows

            // Set camera target (will be lerped in updateCameraFollow)
            camera.targetX = trackCanvas.width / 2 - farthestAgent.car.x * camera.targetScale;
            camera.targetY = trackCanvas.height / 2 - farthestAgent.car.y * camera.targetScale;
        }

        // --- Update Camera Position (using the follow function) ---
        updateCameraFollow();

        // --- Drawing ---
        drawTrack(); // Redraw track with camera transform

        carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height); // Clear car canvas
        if (camera.active) {
            carCtx.save();
            applyCameraTransform(carCtx); // Apply SAME transform to car canvas
        }

        // Draw active agents only
        for (const agent of rlAgents) {
            agent.draw(carCtx); // RLAgent.draw calls car.draw internally
            // Optionally draw lidar for the leading agent
            if (agent.car === controlledCar) {
                agent.car.drawLidar(carCtx);
            }
        }
        if (camera.active) {
            carCtx.restore();
        }
        // --- End Drawing ---


        updateRLStatus(); // Update UI stats

        // Continue the loop if still testing
        if (isTesting) {
            animationId = requestAnimationFrame(gameLoopRL);
        } else {
            animationId = null; // Ensure stopped if isTesting became false
        }

        // --- Remove the safety net timeout, rely on requestAnimationFrame ---
        // setTimeout(ensureRLAnimationContinues, 1000); // Can cause issues
    }



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

        // In init() function, add:
        currentSpecies = [];
        nextSpeciesId = 0;

        // In startRLSimulation(), add near the beginning:
        currentSpecies = []; // Start fresh species list for a new run
        nextSpeciesId = 0;

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
        clearCanvas(); // Clears both track and car canvas

        if (!camera.active && points.length === 0 && startPoint === null) {
            // If camera is not active (e.g., after init) and track is empty, ensure defaults
            resetCamera(); // Make sure camera is fully reset
        }

        if (points.length === 0 && !isPanning && !isTesting) {
            // No points, not panning, not testing: ensure camera isn't arbitrarily active
            // but allow panning to activate it.
            if (camera.x !== 0 || camera.y !== 0 || camera.scale !== 1) {
                camera.active = true; // Keep active if panned/zoomed
            } else {
                camera.active = false;
            }
        } else if (points.length > 0 || isTesting || isPanning) {
            // If there are points OR we are testing/panning, camera should be potentially active
            camera.active = true;
        }


        trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height); // Clear only track canvas here

        if (camera.active) {
            trackCtx.save();
            applyCameraTransform(trackCtx); // Apply camera transform to track context
        }

        // --- Draw track path --- (existing code)
        if (points.length > 0) {
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
        }
        // --- Draw start point --- (existing code)
        if (startPoint) {
            trackCtx.beginPath();
            trackCtx.arc(startPoint.x, startPoint.y, startPointRadius, 0, Math.PI * 2);
            trackCtx.fillStyle = startPointColor;
            trackCtx.fill();
        }
        // --- Draw end point --- (existing code)
        if (points.length > 1) {
            const lastPoint = points[points.length - 1];
            const isCircuit = startPoint &&
                Math.hypot(lastPoint.x - startPoint.x, lastPoint.y - startPoint.y) < pointConnectionThreshold;

            if (!isCircuit) {
                trackCtx.beginPath();
                trackCtx.arc(lastPoint.x, lastPoint.y, startPointRadius, 0, Math.PI * 2);
                trackCtx.fillStyle = finishPointColor;
                trackCtx.fill();
            }

            // --- Update status based on track completion --- (existing code)
            if (!isDrawing) { // Only update status when not actively drawing
                if (isCircuit) {
                    updateStatus('Circuit Track Completed');
                    canvasContainer.classList.add('circuit-track');
                    canvasContainer.classList.remove('finish-track');
                } else {
                    updateStatus('Open Track with Finish Point');
                    canvasContainer.classList.add('finish-track');
                    canvasContainer.classList.remove('circuit-track');
                }
            } else {
                updateStatus('Drawing track...'); // Status while drawing
                canvasContainer.classList.remove('circuit-track', 'finish-track');
            }
        } else if (points.length === 1 && startPoint) {
            updateStatus('Drawing track... Started at start point.');
            canvasContainer.classList.remove('circuit-track', 'finish-track');
        } else if (!isTesting) {
            updateStatus('Draw a track...'); // Default status
            canvasContainer.classList.remove('circuit-track', 'finish-track');
        }


        if (camera.active) {
            trackCtx.restore(); // Restore track context if transform was applied
        }

        // --- Separately draw the car canvas ---
        // Car canvas is cleared and drawn in its respective game loop (gameLoop or gameLoopRL)
    }


    // --- Event listeners for track drawing AND Panning ---

    trackCanvas.addEventListener('contextmenu', e => {
        e.preventDefault(); // Prevent context menu always on canvas
    });

    trackCanvas.addEventListener('mousedown', (e) => {
        if (isTesting) return; // Don't draw or pan during testing

        const rect = trackCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // --- Right Mouse Button for Panning ---
        if (e.button === 2) { // Right mouse button
            if (!isDrawing) { // Don't start panning if already drawing
                isPanning = true;
                panStartX = e.clientX; // Use clientX for delta calculation
                panStartY = e.clientY;
                cameraStartX = camera.x; // Record camera start position
                cameraStartY = camera.y;
                camera.active = true; // Panning activates the camera
                trackCanvas.style.cursor = 'grabbing'; // Change cursor
                document.body.classList.add('panning'); // Add class for potential global styles
                e.preventDefault(); // Prevent default right-click behavior
            }
        }
        // --- Left Mouse Button for Drawing ---
        else if (e.button === 0) { // Left mouse button
            if (!isPanning) { // Don't start drawing if panning
                isDrawing = true;
                trackCanvas.style.cursor = 'crosshair'; // Ensure drawing cursor

                // --- Convert mouse coords to world coords if camera is active ---
                let worldX = mouseX;
                let worldY = mouseY;
                if (camera.active) {
                    // Inverse transform: Translate back from canvas center, unscale, translate back from camera origin
                    worldX = (mouseX - camera.x) / camera.scale;
                    worldY = (mouseY - camera.y) / camera.scale;
                }


                if (points.length === 0) {
                    startPoint = { x: worldX, y: worldY };
                }
                points.push({ x: worldX, y: worldY });
                drawTrack(); // Draw the first point immediately
            }
        }
    });

    // --- Use document for mousemove/mouseup to handle leaving canvas ---
    document.addEventListener('mousemove', (e) => {
        // --- Panning Logic ---
        if (isPanning) {
            const dx = e.clientX - panStartX;
            const dy = e.clientY - panStartY;

            // Update TARGET camera position based on drag delta
            // No scaling needed for delta as it's based on screen pixels
            camera.targetX = cameraStartX + dx;
            camera.targetY = cameraStartY + dy;

            // Directly update camera for immediate feedback (optional, lerp gives smoothness)
            // camera.x = camera.targetX;
            // camera.y = camera.targetY;

            // Let the updateCamera function handle lerping in the game loop if running,
            // otherwise, we need to trigger redraw here.
            // Since we're NOT in a game loop when drawing/panning, call update and draw directly.
            updateCameraPan(); // Use a dedicated pan update or call main one
            drawTrack(); // Redraw track with new camera view
            e.preventDefault(); // Prevent text selection during drag
        }
        // --- Drawing Logic ---
        else if (isDrawing) {
            const rect = trackCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            // Convert mouse coords to world coords if camera is active
            let worldX = mouseX;
            let worldY = mouseY;
            if (camera.active) {
                worldX = (mouseX - camera.x) / camera.scale;
                worldY = (mouseY - camera.y) / camera.scale;
            }


            // Optional: Add distance check to avoid too many points close together
            const lastP = points[points.length - 1];
            if (!lastP || Math.hypot(worldX - lastP.x, worldY - lastP.y) > 5) { // Only add point if moved enough
                points.push({ x: worldX, y: worldY });
                drawTrack();
            }
        }
    });

    document.addEventListener('mouseup', (e) => {
        // --- Stop Panning ---
        if (isPanning && e.button === 2) { // Check button to avoid conflicts if needed
            isPanning = false;
            trackCanvas.style.cursor = 'crosshair'; // Revert cursor
            document.body.classList.remove('panning');
            // Camera remains active, position is where it was left
            // Final checkTrackCompletion might be needed if panning revealed connection
            checkTrackCompletion();
        }
        // --- Stop Drawing ---
        else if (isDrawing && e.button === 0) {
            isDrawing = false;
            trackCanvas.style.cursor = 'crosshair'; // Keep crosshair if not panning
            checkTrackCompletion(); // Check if track closed on mouseup
            drawTrack(); // Final draw to update status correctly
        }
    });

    // --- Update mouseleave for canvas if needed ---
    trackCanvas.addEventListener('mouseleave', () => {
        // Stop drawing if mouse leaves canvas, but DON'T stop panning
        if (isDrawing && !isPanning) {
            isDrawing = false;
            checkTrackCompletion();
            drawTrack(); // Update status
        }
    });

    // --- NEW: Undo Function ---
    function undoLastPoint() {
        if (isTesting || isDrawing || isPanning) return; // Don't undo during simulation, drawing or panning

        if (points.length > 0) {
            points.pop(); // Remove the last point

            if (points.length === 0) {
                startPoint = null; // If no points left, clear start point
                updateStatus('Track cleared by undo.');
                canvasContainer.classList.remove('circuit-track', 'finish-track');
            } else {
                updateStatus('Last point undone.');
                // Re-check completion status after undo
                checkTrackCompletion(); // This will also redraw
            }
            // Explicit redraw needed if checkTrackCompletion doesn't cover all cases
            drawTrack();
        } else {
            updateStatus('Nothing to undo.');
        }
    }

    function resetCamera() {
        camera.x = 0;
        camera.y = 0;
        camera.scale = 1;
        camera.targetX = 0;
        camera.targetY = 0;
        camera.targetScale = 1;
        camera.active = false; // Reset active state
    }
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
        camera.active = false; // Reset active state
    }

    // Renamed the main updateCamera to avoid conflict, used for car following
    function updateCameraFollow() {
        if (!camera.active) return; // Only update if active

        // Lerp towards target (used for car following and potentially zoom)
        camera.x += (camera.targetX - camera.x) * camera.lerpFactor;
        camera.y += (camera.targetY - camera.y) * camera.lerpFactor;
        camera.scale += (camera.targetScale - camera.scale) * camera.lerpFactor;

        // Update target based on controlled car ONLY if not panning
        if (controlledCar && !isPanning) {
            camera.targetX = trackCanvas.width / 2 - controlledCar.x * camera.targetScale;
            camera.targetY = trackCanvas.height / 2 - controlledCar.y * camera.targetScale;
        } else if (!controlledCar && !isPanning) {
            // If no car and not panning, lerp back to origin? Or stay put? Stay put for now.
            // camera.targetX = 0;
            // camera.targetY = 0;
        }
        // If panning, targetX/Y are set directly in the mousemove handler
    }

    // --- NEW: Dedicated camera update for panning ---
    // This allows immediate feedback during pan drag without waiting for animation frame
    function updateCameraPan() {
        if (!isPanning) return;
        // Apply lerp instantly or just set directly for panning? Let's try direct.
        // camera.x = camera.targetX;
        // camera.y = camera.targetY;
        // Using lerp gives a slightly smoother feel even during pan drag
        camera.x += (camera.targetX - camera.x) * camera.lerpFactor * 2; // Faster lerp for panning
        camera.y += (camera.targetY - camera.y) * camera.lerpFactor * 2;

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
        // 1. Translate the canvas origin to where the camera's top-left corner should be
        ctx.translate(camera.x, camera.y);

        // 2. Scale the canvas relative to that new origin
        ctx.scale(camera.scale, camera.scale);

        // Now, drawing at world coordinate (wx, wy) will appear at
        // screen coordinate (camera.x + wx * camera.scale, camera.y + wy * camera.scale)
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
        if (!isTesting || isPanning) { // Pause game loop if panning during testing (unlikely but safe)
            animationId = requestAnimationFrame(gameLoop); // Still request frame to resume
            return;
        }
        const tempCtx = createCollisionCanvas();

        for (const car of cars) {
            car.update(points, startPoint, tempCtx);
        }
        if (controlledCar) {
            updateDistanceCounter(controlledCar.distanceTraveled);
        }

        updateCameraFollow(); // Use the following update here

        // --- Drawing ---
        drawTrack(); // Redraws track with camera transform

        carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height); // Clear car canvas
        if (camera.active) {
            carCtx.save();
            applyCameraTransform(carCtx); // Apply SAME transform to car canvas
        }
        for (const car of cars) {
            car.draw(carCtx);
            // Optionally draw lidar for the controlled car during testing
            if (car === controlledCar) {
                car.drawLidar(carCtx);
            }
        }
        if (camera.active) {
            carCtx.restore();
        }
        // --- End Drawing ---

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
    undoBtn.addEventListener('click', undoLastPoint); // Add listener for Undo

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