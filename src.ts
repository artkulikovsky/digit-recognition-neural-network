// Architecture:
// Images are 28x28 pixels, flattened to 784x1 vector
// Input: 784 → Hidden1: 128 → Hidden2: 64 → Output: 10
// Activation: sigmoid (hidden layers), softmax (output layer)
// Loss: cross-entropy

// Xavier Initialization - used to prevent vanishing/exploding gradients
// W[l] ~ Uniform(-limit, +limit)                  // uniform distribution (random values between -limit and +limit)
// limit = sqrt(6 / (n_in + n_out))                // n_in: number of inputs to layer l, n_out: number of outputs
// b[l] = 0                                        // biases initialized to zero (0)

// Forward Pass:
// a[0] = x                                        // input vector (e.g., shape 784×1)
// for l = 1 to L:
//     z[l] = W[l] @ a[l-1] + b[l]                 // linear combination (matrix-vector product + bias)
//     a[l] = sigmoid(z[l])                        // apply activation function hidden layers
//     a[l] = softmax(z[l])                        // apply activation function output layer

// Sigmoid Function:
// sigmoid(x) = 1 / (1 + exp(-x))                  // activation function
// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))     // efficient derivative form

// Softmax Function:
// softmax(x) = exp(x) / sum(exp(x))               // activation function
// softmax'(x) = softmax(x) * (1 - softmax(x))     // efficient derivative form

// Loss (cross-entropy):
// L = -sum(y * log(a[L]))                         // scalar loss value (output layer is a[L])

// Cost - sum of all losses

// Backpropagation - compute δ[L] for the output layer:
// for softmax + CE: δ[L] = a[L] - y

// Backpropagation - hidden layers (from L-1 down to 1):
// for l = L−1 to 1:
//     dσ_dz = a[l] * (1 - a[l])                   // sigmoid derivative
//     δ[l] = (W[l+1].T @ δ[l+1]) * dσ_dz          // backpropagate error

// Gradients:
// for l = 1 to L:
//     dW[l] = δ[l] @ a[l-1].T                     // outer product of error and previous activation
//     db[l] = δ[l]                                // bias gradient is the error term

// Weight Update (Gradient Descent):
// for l = 1 to L:
//     W[l] -= learning_rate * dW[l]               // update weights
//     b[l] -= learning_rate * db[l]               // update biases

import fs from "fs";

const batchSize = 128;
const learningRate = 0.1;
const neuronsPerLayer = [784, 128, 64, 10];

const sumVectors = (vectors: number[][]): number[] =>
  vectors.reduce((acc, vec) => acc.map((val, i) => val + vec[i]));

const subtractVectors = (vector1: number[], vector2: number[]): number[] =>
  vector1.map((value, index) => value - vector2[index]);

const multiplyVectors = (vector1: number[], vector2: number[]): number[] =>
  vector1.map((value, index) => value * vector2[index]);

const multiplyVectorByScalar = (vector: number[], scalar: number): number[] =>
  vector.map((value) => value * scalar);

const divideVectorByScalar = (vector: number[], scalar: number): number[] =>
  vector.map((val) => val / scalar);

const subtractVectorFromScalar = (scalar: number, vector: number[]) =>
  vector.map((value) => scalar - value);

const transposeMatrix = (matrix: number[][]): number[][] =>
  matrix[0].map((_, index) => matrix.map((row) => row[index]));

const multiplyMatrixByVector = (matrix: number[][], vector: number[]): number[] =>
  matrix.map((row) => row.reduce((acc, value, index) => acc + value * vector[index], 0));

const multiplyMatrixByScalar = (matrix: number[][], scalar: number): number[][] =>
  matrix.map(row => row.map(val => val * scalar));

const outerMatrixProduct = (lineVector: number[], columnVector: number[]): number[][] =>
  lineVector.map((value1) => columnVector.map((value2) => value1 * value2));

const sumMatrices = (matrices: number[][][]): number[][] =>
  matrices.reduce((acc, mat) => acc.map((row, i) => row.map((val, j) => val + mat[i][j])));

const subtractMatrices = (matrix1: number[][], matrix2: number[][]): number[][] =>
  matrix1.map((row, i) => row.map((val, j) => val - matrix2[i][j]));

const divideMatrixByScalar = (matrix: number[][], scalar: number): number[][] =>
  matrix.map(row => row.map(val => val / scalar));

const xavierWeight = (n_in: number, n_out: number) => {
  const limit = Math.sqrt(6 / (n_in + n_out));
  return Math.random() * 2 * limit - limit;
};

const xavierWeightsMatrix = (n_in: number, n_out: number) =>
  new Array(n_out).fill(null).map(() => new Array(n_in).fill(null).map(() => xavierWeight(n_in, n_out)));

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const softmax  = (z: number[]): number[] => {
  const max = Math.max(...z);
  const exps = z.map(v => Math.exp(v - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
};

const crossEntropy = (a: number[], y: number[], ε: number = 1e-9) =>
  -y.reduce((acc, yk, k) => acc + yk * Math.log(a[k] + ε), 0);

const cost = (dataset: [number[], number[]][]) =>
  dataset.reduce((acc, [y, x]) => acc + crossEntropy(forwardPass(x).aL3, y), 0) / dataset.length;

const readJSON = (file: string) => JSON.parse(fs.readFileSync(`${file}.json`, "utf8"));
const writeJSON = (file: string, data: any) => fs.writeFileSync(`${file}.json`, JSON.stringify(data));

let testData: [number[], number[]][] = readJSON("test_data");
let trainData: [number[], number[]][] = readJSON("train_data");
let parameters: { bL1: number[], bL2: number[], bL3: number[], wL1: number[][], wL2: number[][], wL3: number[][] } = readJSON("parameters");

const resetParameters = () => {
  const bL1 = new Array(neuronsPerLayer[1]).fill(0);
  const bL2 = new Array(neuronsPerLayer[2]).fill(0);
  const bL3 = new Array(neuronsPerLayer[3]).fill(0);
  const wL1 = xavierWeightsMatrix(neuronsPerLayer[0], neuronsPerLayer[1]);
  const wL2 = xavierWeightsMatrix(neuronsPerLayer[1], neuronsPerLayer[2]);
  const wL3 = xavierWeightsMatrix(neuronsPerLayer[2], neuronsPerLayer[3]);

  writeJSON("parameters", { bL1, bL2, bL3, wL1, wL2, wL3 }); parameters = { bL1, bL2, bL3, wL1, wL2, wL3 };
};

const evaluateNetwork = () => {
  let correct = 0;

  testData.forEach(([y, aL0]) => {
    const { aL3 } = forwardPass(aL0);
    if (aL3.indexOf(Math.max(...aL3)) === y.indexOf(Math.max(...y))) correct++;
  });

  console.log(`Accuracy: ${Number((correct / testData.length * 100).toFixed(2))}%`);
};

const forwardPass = (aL0: number[]): { aL1: number[], aL2: number[], aL3: number[] } => {
  const { bL1, bL2, bL3, wL1, wL2, wL3 } = parameters;

  const zL1 = sumVectors([multiplyMatrixByVector(wL1, aL0), bL1]);
  const aL1 = zL1.map(sigmoid);

  const zL2 = sumVectors([multiplyMatrixByVector(wL2, aL1), bL2]);
  const aL2 = zL2.map(sigmoid);

  const zL3 = sumVectors([multiplyMatrixByVector(wL3, aL2), bL3]);
  const aL3 = softmax(zL3);

  return { aL1, aL2, aL3 };
};

const backpropagation = (aL1: number[], aL2: number[], aL3: number[], y: number[]) => {
  const { wL2, wL3 } = parameters;

  // Output Layer: δ[L] = a[L] - y
  const δL3 = subtractVectors(aL3, y);

  // Hidden Layer 2: δ[2] = (W[3].T @ δ[3]) * (a[2] * (1 - a[2]))
  const dSig_dzL2 = multiplyVectors(aL2, subtractVectorFromScalar(1, aL2));
  const δL2 = multiplyVectors(multiplyMatrixByVector(transposeMatrix(wL3), δL3), dSig_dzL2);

  // Hidden Layer 1: // δ[1] = (W[2].T @ δ[2]) * (a[1] * (1 - a[1]))
  const dSig_dzL1 = multiplyVectors(aL1, subtractVectorFromScalar(1, aL1));
  const δL1 = multiplyVectors(multiplyMatrixByVector(transposeMatrix(wL2), δL2), dSig_dzL1);

  return { δL1, δL2, δL3 };
};

const gradient = (aL0: number[], aL1: number[], aL2: number[], δL1: number[], δL2: number[], δL3: number[]) => {
  // Gradient of biases: db[l] = δ[l]
  const db_L1 = δL1;
  const db_L2 = δL2;
  const db_L3 = δL3;

  // Gradient of weights: dW[l] = δ[l] @ a[l-1].T
  const dW_L1 = outerMatrixProduct(δL1, aL0);
  const dW_L2 = outerMatrixProduct(δL2, aL1);
  const dW_L3 = outerMatrixProduct(δL3, aL2);

  return { dW_L1, dW_L2, dW_L3, db_L1, db_L2, db_L3 };
}

const trainNetwork = () => {
  console.log('Starting epoch, batch size:', batchSize, 'total batches:', Math.ceil(trainData.length / batchSize));

  for (let i = 0; i < trainData.length; i += batchSize) {
    const batch = trainData.slice(i, i + batchSize);
    const gradients: { dW_L1: number[][], dW_L2: number[][], dW_L3: number[][], db_L1: number[], db_L2: number[], db_L3: number[] }[] = [];

    // Average gradient of biases: db_avg = db_sum[l] / batch_size
    const db_avg = (key: 'db_L1' | 'db_L2' | 'db_L3') => divideVectorByScalar(sumVectors(gradients.map(g => g[key])), batch.length);

    // Average gradient of weights: dW_avg = dW_sum[l] / batch_size
    const dW_avg = (key: 'dW_L1' | 'dW_L2' | 'dW_L3') => divideMatrixByScalar(sumMatrices(gradients.map(g => g[key])), batch.length);

    // Updated biases: b[l] -= learning_rate * db[l]
    const bL = (key: 'bL1' | 'bL2' | 'bL3', db_L_avg: number[]) => subtractVectors(parameters[key], multiplyVectorByScalar(db_L_avg, learningRate));

    // Updated weights: W[l] -= learning_rate * dW[l]
    const wL = (key: 'wL1' | 'wL2' | 'wL3', dW_L_avg: number[][]) => subtractMatrices(parameters[key], multiplyMatrixByScalar(dW_L_avg, learningRate));
    
    batch.forEach(([y, aL0]) => {
      const { aL1, aL2, aL3 } = forwardPass(aL0);
      const { δL1, δL2, δL3 } = backpropagation(aL1, aL2, aL3, y);
      gradients.push(gradient(aL0, aL1, aL2, δL1, δL2, δL3))
    });

    const [db_L1_avg, db_L2_avg, db_L3_avg] = [db_avg('db_L1'), db_avg('db_L2'), db_avg('db_L3')];
    const [dW_L1_avg, dW_L2_avg, dW_L3_avg]  = [dW_avg('dW_L1'), dW_avg('dW_L2'), dW_avg('dW_L3')];

    const [bL1, bL2, bL3] = [bL('bL1', db_L1_avg), bL('bL2', db_L2_avg), bL('bL3', db_L3_avg)];
    const [wL1, wL2, wL3] = [wL('wL1', dW_L1_avg), wL('wL2', dW_L2_avg), wL('wL3', dW_L3_avg)];

    const updatedParameters = { bL1, bL2, bL3, wL1, wL2, wL3 };

    writeJSON("parameters", updatedParameters); parameters = updatedParameters;

    console.log('Batch #', i / batchSize + 1, 'average cost:', cost(batch));
  }
};

// Run one of the following functions to execute the corresponding task
// resetParameters();
// trainNetwork();
// evaluateNetwork();
