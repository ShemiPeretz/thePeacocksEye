import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Mock Coordinates for 3 locations in Israel
data = {
    'lat': [31.7683, 32.0853, 32.7940, 31.2573139707],  # Israel latitude values
    'lon': [35.2137, 34.7818, 34.9896, 34.800903463],  # Israel longitude values
    'temperature': [20, 25, 15, 30]  # Sample temperature values
}

df = pd.DataFrame(data)

fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="temperature",
                        size_max=15, zoom=1, mapbox_style="carto-positron")

# Update the traces to use pin symbols and display temperature values
fig.update_traces(marker=go.scattermapbox.Marker(size=14, color=df['temperature'], colorscale="reds", opacity=0.8))

beer_sheva_coords = {'lat': 31.2573139707, 'lon': 34.800903463}

# Update the layout to use Mapbox
# fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=6, mapbox_center=beer_sheva_coords)
fig.update_layout(mapbox_style='carto-positron', mapbox_zoom=7, mapbox_center=beer_sheva_coords)

# Update the layout to hide the color bar
fig.update_layout(coloraxis_showscale=False, margin={"r":0,"t":0,"l":0,"b":0})

fig.show()

fig.write_html("C:\\Users\\shemi\\thePeacocksEyeClient\\src\\assets\\map\\map.html")
