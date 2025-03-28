# RL Track Designer & Neural Evolution Simulator


*README ini available dalam Bahasa Indonesia dan English / This README is available in Indonesian and English*

## 🇮🇩 Deskripsi

Aplikasi ini adalah *simulation* evolusi *neural network* untuk training *self-driving cars* menggunakan *NEAT (NeuroEvolution of Augmenting Topologies) algorithm*. User dapat menggambar *track* dengan berbagai bentuk, dan *car simulation* akan try to learn cara terbaik untuk menyelesaikan *track* tersebut.

Program ini menggunakan teknik *neural networks* dan *genetic algorithms* untuk mengoptimalkan *performance* mobil melalui beberapa *generations*. Seiring waktu, mobil akan *learn* untuk avoid *obstacles* dan menyelesaikan *track* dengan lebih *efficiently*.

## 🇺🇸 Description

This application is a neural network evolution simulation for training self-driving cars using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. Users can draw tracks of various shapes, and the car simulation will try to learn the best way to complete the track.

The program uses neural networks and genetic algorithms to optimize car performance over several generations. Over time, cars will learn to avoid obstacles and complete the track more efficiently.

## 🛠️ Features / Fitur-Fitur

### 🇮🇩 Features
- *Interactive track designer* dengan *undo functionality*.
- *Neural network-controlled car simulation*.
- Implementasi *NEAT algorithm* untuk *neural network evolution*.
- *Speciation* untuk maintain *genetic diversity*.
- *Best network visualization*.
- *Export/import trained models* dan *tracks*.
- *Pan and zoom camera* untuk view *simulations* dengan lebih jelas.

### 🇺🇸 Features
- Interactive track designer with undo functionality
- Neural network-controlled car simulation
- NEAT algorithm implementation for neural network evolution
- Speciation to maintain genetic diversity
- Best network visualization
- Export/import of trained models and tracks
- Pan and zoom camera to view simulations more clearly

## 🚀 Cara Menggunakan / How to Use

### 🇮🇩 How to Use
1.  **Drawing Tracks**:
    -   Klik dan *drag* di *canvas* untuk menggambar *track*.
    -   Klik kanan dan *drag* untuk *pan view*.
    -   Gunakan tombol "*Undo Last Point*" untuk remove *point* terakhir.
    -   Hubungkan *track* kembali ke *starting point* untuk membuat *closed circuit*.

2.  **Testing Tracks**:
    -   Klik "*Test Track*" untuk test *track* dengan *manual controls*.
    -   Gunakan *arrow keys* atau WASD untuk mengendalikan mobil.

3.  **RL Learning Mode**:
    -   Klik "*Run RL*" untuk start *learning simulation*.
    -   Mobil akan *evolve* melalui beberapa *generations*.
    -   *Network visualization panel* akan show struktur *network* terbaik.

4.  **Export/Import**:
    -   Gunakan tombol "*Export Track*" untuk save *created tracks*.
    -   Gunakan tombol "*Import Track*" untuk load *saved tracks*.
    -   Gunakan tombol "*Export Best Model*" untuk save *best model*.

### 🇺🇸 How to Use
1.  **Drawing Tracks**:
    -   Click and drag on the canvas to draw a track
    -   Right-click and drag to pan the view
    -   Use the "Undo Last Point" button to remove the last point
    -   Connect the track back to the starting point to create a closed circuit

2.  **Testing Tracks**:
    -   Click "Test Track" to test the track with manual controls
    -   Use arrow keys or WASD to control the car

3.  **RL Learning Mode**:
    -   Click "Run RL" to start the learning simulation
    -   Cars will evolve through several generations
    -   The network visualization panel will show the best network structure

4.  **Export/Import**:
    -   Use "Export Track" button to save your created tracks
    -   Use "Import Track" button to load saved tracks
    -   Use "Export Best Model" button to save the best model

## 🧠 Teknologi / Technologies

### 🇮🇩 Technologies
-   **JavaScript** - *Primary programming language*.
-   **HTML5 Canvas** - Untuk *track* dan *simulation rendering*.
-   **NEAT** - *NeuroEvolution of Augmenting Topologies* untuk *neural network evolution*.
-   **Lidar Sensors** - *Simulated sensors* untuk detect jarak ke *obstacles*.

### 🇺🇸 Technologies
-   **JavaScript** - Primary programming language
-   **HTML5 Canvas** - For track and simulation rendering
-   **NEAT** - NeuroEvolution of Augmenting Topologies for neural network evolution
-   **Lidar Sensors** - Simulated sensors to detect distance to obstacles

## 🤔 Cara Kerja / How It Works

### 🇮🇩 How It Works
Aplikasi ini menggunakan *NEAT algorithm* yang menggabungkan evolusi *neural network topology* dengan *genetic algorithms*:

1.  **Input**: *Lidar sensor readings*, *car speed*, dan *direction*.
2.  **Processing**: *Neural networks* memproses *inputs* dan decide *actions*.
3.  **Output**: *Steering controls* (kiri, kanan, maju, mundur).
4.  **Evaluation**: Mobil dievaluasi berdasarkan distance traveled on *track*.
5.  **Evolution**: *Best networks* dipertahankan dan *evolved* ke *next generation*.
6.  **Speciation**: *Networks* di-*group* untuk maintain *genetic diversity*.

### 🇺🇸 How It Works
This application uses the NEAT algorithm which combines neural network topology evolution with genetic algorithms:

1.  **Input**: Lidar sensor readings, car speed, and direction
2.  **Processing**: Neural networks process inputs and decide actions
3.  **Output**: Steering controls (left, right, forward, backward)
4.  **Evaluation**: Cars are evaluated based on distance traveled on track
5.  **Evolution**: Best networks are kept and evolved to next generation
6.  **Speciation**: Networks are grouped to maintain genetic diversity

## ⚙️ Installation / Instalasi

### 🇮🇩 Installation
1.  *Clone repository* ini.
2.  Buka file `index.html` di *web browser*.
3.  Atau run dengan *local server* (*recommended*): `node server.js`.

### 🇺🇸 Installation
1.  Clone this repository
2.  Open `index.html` in a web browser
3.  Or run with a local server (recommended): `node server.js`

## 🤝 Kontribusi / Contributing

*Contributions* sangat *welcome*!

## 📄 Lisensi / License

[MIT License](LICENSE)
