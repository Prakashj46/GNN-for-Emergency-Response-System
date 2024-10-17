# Multi-Criteria-based Graph Neural Networks for Emergency Response System

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Technology Stack](#technology-stack)
- [Graph Neural Networks (GNN) Architecture](#graph-neural-networks-gnn-architecture)
- [Dataset](#dataset)
- [Challenges and Solutions](#challenges-and-solutions)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Project Overview
This project aims to build an **Emergency Response System** using **Graph Neural Networks (GNN)** that leverages multiple criteria such as location, severity, resource availability, traffic conditions, and medical service response times. The system is designed to recommend the optimal response to medical emergencies by considering these criteria.

## Motivation
During critical medical emergencies, response times are essential for saving lives. Traditional systems do not account for multiple factors that impact emergency response times, such as traffic, resource availability, and the severity of the situation. By using a graph-based approach with GNN, we can optimize and speed up emergency response decisions by considering these various factors collectively.

## Objectives
- To build a scalable and efficient medical emergency response system.
- To use **Graph Neural Networks (GNN)** to process complex interrelations between multiple criteria.
- To predict the optimal emergency response route based on factors like location, traffic, available resources, and severity.

## Key Features
- **Multi-Criteria Decision Making**: Combines factors such as location, traffic, severity, and resources for optimal decision-making.
- **Graph-Based Model**: Uses Graph Neural Networks (GNN) to process relationships between entities (hospitals, traffic points, resources).
- **Real-Time Updates**: Considers dynamic data inputs like traffic conditions and resource availability.
- **Scalability**: Designed to be scaled for large urban environments with complex traffic and resource networks.

## Methodology
The project follows the steps outlined below:

1. **Data Collection**: 
    - Gathered historical data for emergency medical services, traffic data, and resource availability from multiple sources.
2. **Graph Construction**: 
    - Created a graph where nodes represent hospitals, traffic junctions, and resources, and edges represent possible routes with weights based on criteria such as time, distance, and severity.
3. **Graph Neural Network (GNN)**:
    - Designed a GNN to propagate information between nodes (hospitals, traffic points, etc.) to predict the optimal emergency response.
4. **Training**: 
    - Trained the GNN on historical data to optimize for minimum response time and optimal resource allocation.
5. **Prediction and Optimization**:
    - The trained model is used to predict the best route and resource for each emergency event.

## Technology Stack
- **Programming Language**: Python
- **Machine Learning Framework**: PyTorch (for GNN implementation)
- **Graph Libraries**: PyTorch Geometric
- **Data Processing**: Pandas, NumPy
- **APIs for Traffic & Weather**: Google Maps API, OpenWeather API
- **Database**: SQLite (for storing historical data)
- **Visualization**: Matplotlib, NetworkX (for visualizing graphs)

## Graph Neural Networks (GNN) Architecture
The GNN architecture is designed to propagate features between nodes (hospitals, traffic points, resources) and update the edge weights (representing routes) accordingly. Key components of the architecture include:
- **Graph Convolutional Layers (GCN)**: Propagate node-level features across connected nodes.
- **Edge Weight Update Mechanism**: Dynamically adjust edge weights based on real-time inputs like traffic and resource availability.
- **Softmax Classifier**: Used for predicting the best emergency response route.

## Dataset
- **Hospital Locations**: Geographic coordinates and emergency handling capacity of hospitals.
- **Traffic Data**: Real-time traffic data collected from Google Maps API.
- **Resource Data**: Availability of medical resources like ambulances, doctors, and medical supplies.
- **Emergency Records**: Historical records of medical emergencies, response times, and outcomes.

## Challenges and Solutions
- **Real-Time Data Integration**: One of the challenges was integrating real-time traffic data into the model. This was solved by using Google Maps API, which provided live updates that could be processed by the GNN in real-time.
- **Graph Construction**: Building an efficient graph to represent all criteria without overloading the model was a challenge. This was managed by optimizing node and edge connections based on priority criteria such as severity and distance.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gnn-emergency-response-system.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up API keys for Google Maps :
    - Create a `.env` file and add your API keys:
      ```bash
      GOOGLE_MAPS_API_KEY=your_google_maps_api_key
      ```

4. Run the project:
    ```bash
    python main.py
    ```

## Usage
1. Start the system using the command:
   ```bash
   python main.py
