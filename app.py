from flask import Flask, render_template

app = Flask(__name__)

# Paste your Python code here
import pandas as pd
import numpy as np
import folium
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from math import radians
import requests

# Step 1: Generate a synthetic dataset of hospitals with various criteria
data = {
    'Hospital Name': ['Hospital A', 'Hospital B', 'Hospital C', 'Hospital D', 'Hospital E', 'Hospital F'],
    'Latitude': [10.8100, 11.0168, 9.9173, 10.8090, 10.8080, 10.8110],
    'Longitude': [78.6800, 76.9558, 78.1199, 78.6810, 78.6820, 78.6790],
    'Doctors Total': [50, 60, 70, 50, 60, 70],
    'Doctors Available': [30, 40, 50, 30, 30, 50],
    'Ambulances Total': [10, 15, 20, 30, 15, 19],
    'Ambulances Available': [5, 10, 15, 12, 9, 10],
    'Beds Total': [100, 120, 150, 100, 110, 120],
    'Beds Occupied': [70, 80, 90, 80, 90, 100],
    'Beds Available': [30, 40, 60, 20, 20, 20]
}

hospital_df = pd.DataFrame(data)


# Step 2: Simulate an accident location
accident_location = {'Latitude': 10.304475, 'Longitude': 77.939910}
num_injured = 8  # Simulating 8 injured people

# Step 3: Calculate distances between the accident location and hospitals
def haversine_vectorize(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c  # Radius of the Earth in kilometers
    return km

hospital_df['Distance'] = haversine_vectorize(
    hospital_df['Longitude'],
    hospital_df['Latitude'],
    accident_location['Longitude'],
    accident_location['Latitude']
)

# Step 4: Define the Graph Neural Network model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

# Step 5: Prepare data for training the GNN
X = hospital_df[['Distance', 'Doctors Available', 'Ambulances Available', 'Beds Available']].values
y = (hospital_df['Distance'] <= 10).astype(int).values  # Binary label: 1 if hospital is within 10 km, 0 otherwise

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to match the model output shape

# Step 6: Define training parameters and train the GNN
input_dim = X.shape[1]
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 1

model = GNN(input_dim, hidden_dim1, hidden_dim2, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(2000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Step 7: Predict hospital availability using the trained model
hospital_df['Availability Prediction'] = model(X).detach().numpy()

# Step 8: Visualize the results using Folium
hospital_map = folium.Map(location=[accident_location['Latitude'], accident_location['Longitude']], zoom_start=12)

# Add accident location marker
folium.Marker(
    location=[accident_location['Latitude'], accident_location['Longitude']],
    popup=f"Accident Location\nInjured: {num_injured}",
    icon=folium.Icon(color='red', icon='exclamation-triangle')
).add_to(hospital_map)

# Add hospital markers
for _, hospital in hospital_df.iterrows():
    popup_text = f"{hospital['Hospital Name']}\nDistance: {hospital['Distance']:.2f} km\nAvailability: {hospital['Availability Prediction']:.2f}"
    if hospital['Availability Prediction'] >= 0.5:
        icon_color = 'green'
    else:
        icon_color = 'blue'  # All hospitals will be in blue color
    folium.Marker(
        location=[hospital['Latitude'], hospital['Longitude']],
        popup=popup_text,
        icon=folium.Icon(color=icon_color, icon='hospital')
    ).add_to(hospital_map)

# Find the nearest available hospital
nearest_hospital_idx = hospital_df['Availability Prediction'].idxmax()
nearest_hospital = hospital_df.loc[nearest_hospital_idx]

# Function to get the navigation route using OpenRouteService API
def get_navigation_route(start_coords, end_coords, api_key):
    url = f'https://api.openrouteservice.org/v2/directions/driving-car?api_key={api_key}&start={start_coords[1]},{start_coords[0]}&end={end_coords[1]},{end_coords[0]}'
    response = requests.get(url)
    route_data = response.json()
    geometry = route_data['features'][0]['geometry']
    coordinates = geometry['coordinates']
    return coordinates

# Define your OpenRouteService API key
openrouteservice_api_key = '5b3ce3597851110001cf6248ebdff5bbe50f430abb315a1f41d7bf52'

# Find coordinates of the accident location and nearest hospital
accident_coords = (accident_location['Latitude'], accident_location['Longitude'])
nearest_hospital_coords = (nearest_hospital['Latitude'], nearest_hospital['Longitude'])

# Get the navigation route
navigation_route = get_navigation_route(accident_coords, nearest_hospital_coords, openrouteservice_api_key)

# Add navigation route to the map
folium.PolyLine(locations=[(lat, lon) for lon, lat in navigation_route], color="blue", weight=5, opacity=0.7).add_to(hospital_map)

# Save the map to an HTML file
hospital_map.save('hospital_map_with_gnn_and_route.html')

print("Navigation route added to the map.")

# Print the nearest available hospital
print("Accident Location:", accident_location)
print(f"Number of Injured: {num_injured}")
print("Nearest Available Hospital:")
print(nearest_hospital)

@app.route('/')
def index():
    return render_template('map.html', nearest_hospital=nearest_hospital)

if __name__ == '__main__':
    app.run(debug=True)
