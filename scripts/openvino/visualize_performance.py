import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Get CSV filename from command-line argument
if len(sys.argv) < 2:
    print("Usage: python visualize_performance.py <csv_filename>")
    print("Example: python visualize_performance.py benchmarking_results_2026-05-05_16-52.csv")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found!")
    sys.exit(1)

# Read the CSV file
df = pd.read_csv(csv_file)

# Filter out failed tests
df = df[df['Device'] != 'FAILED']
df = df[df['TTFT (ms)'] != 'FAILED']

# Convert numeric columns to float
numeric_cols = ['TTFT (ms)', 'PROMPT_EVAL / PP (tok/s)', 'TOKEN_GEN (tok/s)', 
                'AVG_TOKEN_LATENCY (ms/tok)', 'GENERATION_TIME (s)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NaN values in key metrics
df = df.dropna(subset=['TTFT (ms)', 'TOKEN_GEN (tok/s)'])

# Define framework order
framework_order = [
    'llamaCPP_ggml_CPU',
    'llamaCPP_OV_CPU',
    'llamaCPP_OV_GPU',
    'llamaCPP_Vulkan_GPU',
    'llamaCPP_OV_NPU',
    'OV_GENAI_GGUF_CPU_ic32',
    'OV_GENAI_GGUF_CPU_ic128',
    'OV_GENAI_GGUF_GPU_ic32',
    'OV_GENAI_GGUF_GPU_ic128',
    'OV_GENAI_IR_CPU_DEFAULT_ic32',
    'OV_GENAI_IR_CPU_DEFAULT_ic128',
    'OV_GENAI_IR_CPU_CW_ic32',
    'OV_GENAI_IR_CPU_CW_ic128',
    'OV_GENAI_IR_CPU_GS32_ic32',
    'OV_GENAI_IR_CPU_GS32_ic128',
    'OV_GENAI_IR_GPU_DEFAULT_ic32',
    'OV_GENAI_IR_GPU_DEFAULT_ic128',
    'OV_GENAI_IR_GPU_CW_ic32',
    'OV_GENAI_IR_GPU_CW_ic128',
    'OV_GENAI_IR_GPU_GS32_ic32',
    'OV_GENAI_IR_GPU_GS32_ic128',
    'OV_GENAI_IR_NPU_CW_ic32',
    'OV_GENAI_IR_NPU_CW_ic128',
    'OV_GENAI_IR_NPU_GS32_ic32',
    'OV_GENAI_IR_NPU_GS32_ic128'
]

# Color palette for frameworks
color_palette = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
    '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400',
    '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#16a085'
]

# Create framework color mapping
framework_colors = {}
all_frameworks = df['Framework'].unique()
for fw in all_frameworks:
    if fw in framework_order:
        idx = framework_order.index(fw)
    else:
        idx = len(framework_order) + list(all_frameworks).index(fw)
    framework_colors[fw] = color_palette[idx % len(color_palette)]


# Function to create charts for each device
def create_device_charts(device, device_data, device_color):
    """Create TTFT and Token Generation charts for a specific device using Plotly"""
    
    # Group by model and framework
    models = sorted(device_data['Model'].unique())
    frameworks_in_device = device_data['Framework'].unique()
    
    # Sort frameworks by the defined order
    sorted_frameworks = sorted(frameworks_in_device, 
                               key=lambda x: framework_order.index(x) if x in framework_order else 999)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time to First Token (TTFT)', 'Token Generation Speed'),
        horizontal_spacing=0.1
    )
    
    # Add traces for TTFT
    for fw in sorted_frameworks:
        ttft_values = []
        token_gen_values = []
        model_labels = []
        
        for model in models:
            model_fw_data = device_data[(device_data['Model'] == model) & (device_data['Framework'] == fw)]
            if len(model_fw_data) > 0:
                ttft_values.append(model_fw_data['TTFT (ms)'].values[0])
                token_gen_values.append(model_fw_data['TOKEN_GEN (tok/s)'].values[0])
                model_labels.append(model)
        
        if ttft_values:
            # Add TTFT bars
            fig.add_trace(
                go.Bar(
                    name=fw,
                    x=model_labels,
                    y=ttft_values,
                    marker_color=framework_colors[fw],
                    showlegend=True,
                    legendgroup=fw
                ),
                row=1, col=1
            )
            
            # Add Token Gen bars
            fig.add_trace(
                go.Bar(
                    name=fw,
                    x=model_labels,
                    y=token_gen_values,
                    marker_color=framework_colors[fw],
                    showlegend=False,
                    legendgroup=fw
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="TTFT (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Token Gen (tok/s)", row=1, col=2)
    
    fig.update_layout(
        title_text=f"{device} Performance",
        barmode='group',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        template="plotly_white"
    )
    
    return fig


# Device colors
device_colors = {'CPU': '#3498db', 'GPU': '#e74c3c', 'NPU': '#2ecc71'}

# Create charts for each device and collect them
all_figures = []
for device in ['CPU', 'GPU', 'NPU']:
    device_data = df[df['Device'] == device].copy()
    if len(device_data) > 0:
        fig = create_device_charts(device, device_data, device_colors[device])
        all_figures.append(fig)

# Generate HTML with all charts
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Analysis by Device</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-table th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .stats-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        .stats-table tr:hover {
            background-color: #f5f5f5;
        }
        .device-cpu { color: #3498db; font-weight: bold; }
        .device-gpu { color: #e74c3c; font-weight: bold; }
        .device-npu { color: #2ecc71; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Performance Analysis by Device</h1>
"""

# Add each device chart
for i, fig in enumerate(all_figures):
    html_content += f'    <div class="chart-container" id="chart{i}"></div>\n'

html_content += """
    <script>
"""

# Add plotly chart data
for i, fig in enumerate(all_figures):
    chart_json = fig.to_json()
    html_content += f"        var chartData{i} = {chart_json};\n"
    html_content += f"        Plotly.newPlot('chart{i}', chartData{i}.data, chartData{i}.layout);\n"

html_content += """
    </script>
</body>
</html>
"""

# Save HTML file
output_file = 'visualize_performance.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

# Save HTML file
output_file = 'visualize_performance.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY BY DEVICE")
print("="*80)
for device in df['Device'].unique():
    device_data = df[df['Device'] == device]
    print(f"\n{device} ({len(device_data)} tests):")
    print(f"  TTFT (ms):              {device_data['TTFT (ms)'].mean():.2f} ± {device_data['TTFT (ms)'].std():.2f}")
    print(f"  Token Gen (tok/s):      {device_data['TOKEN_GEN (tok/s)'].mean():.2f} ± {device_data['TOKEN_GEN (tok/s)'].std():.2f}")
    print(f"  Prompt Eval (tok/s):    {device_data['PROMPT_EVAL / PP (tok/s)'].mean():.2f} ± {device_data['PROMPT_EVAL / PP (tok/s)'].std():.2f}")
    print(f"  Avg Latency (ms/tok):   {device_data['AVG_TOKEN_LATENCY (ms/tok)'].mean():.2f} ± {device_data['AVG_TOKEN_LATENCY (ms/tok)'].std():.2f}")
print("="*80)

print(f"\n✓ Interactive HTML visualization saved: {output_file}")
print(f"  Open the file in your browser to view interactive charts!")
