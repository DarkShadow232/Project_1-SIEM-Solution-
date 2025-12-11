/**
 * SIEM Dashboard - Chart Configurations
 * Plotly chart themes and configurations
 */

// Default Plotly theme for dark mode
const plotlyDarkTheme = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        color: '#e0e0e0',
        family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'
    },
    xaxis: {
        gridcolor: '#3a3a3a',
        linecolor: '#3a3a3a',
        tickcolor: '#3a3a3a'
    },
    yaxis: {
        gridcolor: '#3a3a3a',
        linecolor: '#3a3a3a',
        tickcolor: '#3a3a3a'
    }
};

// Color palettes
const colorPalettes = {
    severity: {
        high: '#dc3545',
        medium: '#fd7e14',
        low: '#ffc107'
    },
    detection: {
        iqr: '#00bfa5',
        zscore: '#ff6f00',
        isolation: '#9c27b0'
    },
    status: {
        new: '#dc3545',
        investigating: '#ffc107',
        resolved: '#28a745'
    },
    gradient: ['#00bfa5', '#00d4a0', '#00e89c', '#42f596', '#66ff91']
};

// Create time series chart
function createTimeSeriesChart(elementId, data, title) {
    const trace = {
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: colorPalettes.detection.iqr,
            width: 3
        },
        marker: {
            size: 8,
            color: colorPalettes.detection.iqr
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(0, 191, 165, 0.2)'
    };
    
    const layout = {
        ...plotlyDarkTheme,
        title: title,
        xaxis: {
            ...plotlyDarkTheme.xaxis,
            title: 'Time'
        },
        yaxis: {
            ...plotlyDarkTheme.yaxis,
            title: 'Count'
        },
        margin: {l: 50, r: 20, t: 40, b: 50},
        hovermode: 'closest'
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create bar chart
function createBarChart(elementId, data, title) {
    const trace = {
        x: data.labels,
        y: data.values,
        type: 'bar',
        marker: {
            color: data.colors || colorPalettes.gradient
        }
    };
    
    const layout = {
        ...plotlyDarkTheme,
        title: title,
        xaxis: {
            ...plotlyDarkTheme.xaxis
        },
        yaxis: {
            ...plotlyDarkTheme.yaxis,
            title: 'Count'
        },
        margin: {l: 50, r: 20, t: 40, b: 50}
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create pie chart
function createPieChart(elementId, data, title) {
    const trace = {
        values: data.values,
        labels: data.labels,
        type: 'pie',
        marker: {
            colors: data.colors
        },
        textfont: {
            color: '#000'
        },
        hole: 0.4
    };
    
    const layout = {
        ...plotlyDarkTheme,
        title: title,
        margin: {l: 20, r: 20, t: 40, b: 20},
        showlegend: true,
        legend: {
            orientation: 'v',
            x: 1,
            y: 0.5
        }
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create 3D scatter plot
function create3DScatter(elementId, data, title) {
    const trace = {
        x: data.x,
        y: data.y,
        z: data.z,
        mode: 'markers',
        marker: {
            size: 5,
            color: data.colors,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: data.colorbarTitle || 'Value',
                thickness: 15,
                len: 0.7
            }
        },
        type: 'scatter3d',
        text: data.labels,
        hovertemplate: '<b>%{text}</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
    };
    
    const layout = {
        ...plotlyDarkTheme,
        title: title,
        scene: {
            xaxis: {
                title: data.xAxisTitle || 'X',
                gridcolor: '#3a3a3a',
                backgroundcolor: 'rgba(0,0,0,0)'
            },
            yaxis: {
                title: data.yAxisTitle || 'Y',
                gridcolor: '#3a3a3a',
                backgroundcolor: 'rgba(0,0,0,0)'
            },
            zaxis: {
                title: data.zAxisTitle || 'Z',
                gridcolor: '#3a3a3a',
                backgroundcolor: 'rgba(0,0,0,0)'
            }
        },
        height: 500
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create heatmap
function createHeatmap(elementId, data, title) {
    const trace = {
        z: data.z,
        x: data.x,
        y: data.y,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        text: data.text,
        texttemplate: '%{text}',
        textfont: {
            color: '#fff'
        }
    };
    
    const layout = {
        ...plotlyDarkTheme,
        title: title,
        height: 400
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create gauge chart
function createGauge(elementId, value, title, range = [0, 100]) {
    const trace = {
        type: "indicator",
        mode: "gauge+number",
        value: value,
        title: {
            text: title,
            font: {color: '#e0e0e0'}
        },
        gauge: {
            axis: {
                range: range,
                tickcolor: '#e0e0e0'
            },
            bar: {color: colorPalettes.severity.high},
            bgcolor: "#2d2d2d",
            borderwidth: 2,
            bordercolor: "#3a3a3a",
            steps: [
                {range: [0, 33], color: colorPalettes.status.resolved},
                {range: [33, 66], color: colorPalettes.status.investigating},
                {range: [66, 100], color: colorPalettes.status.new}
            ],
            threshold: {
                line: {color: "white", width: 4},
                thickness: 0.75,
                value: 70
            }
        }
    };
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#e0e0e0'},
        height: 250,
        margin: {l: 20, r: 20, t: 20, b: 20}
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Update chart data dynamically
function updateChart(elementId, newData, traceIndex = 0) {
    Plotly.update(elementId, newData, {}, [traceIndex]);
}

// Resize all charts (call on window resize)
function resizeAllCharts() {
    const chartElements = document.querySelectorAll('[id$="Chart"], [id$="Plot"]');
    chartElements.forEach(element => {
        Plotly.Plots.resize(element);
    });
}

// Window resize handler
window.addEventListener('resize', debounce(resizeAllCharts, 250));

console.log('Chart configurations loaded');

