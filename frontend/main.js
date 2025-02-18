// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const analyzeForm = document.getElementById('analyzeForm');
    const questionInput = document.getElementById('questionInput');
    const databaseSelect = document.getElementById('databaseType');
    const analyzeButton = document.getElementById('analyzeButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorContainer = document.getElementById('errorContainer');

    // Function to set loading state
    function setLoading(isLoading) {
        loadingSpinner.style.display = isLoading ? 'inline-block' : 'none';
        analyzeButton.disabled = isLoading;
        questionInput.disabled = isLoading;
        databaseSelect.disabled = isLoading;
    }

    // Function to show error
    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
        resultsContainer.innerHTML = '';
    }

    // Function to clear results
    function clearResults() {
        resultsContainer.innerHTML = '';
        errorContainer.style.display = 'none';
    }

    // Function to create a section
    function createSection(title, content) {
        const section = document.createElement('div');
        section.className = 'result-section';
        section.innerHTML = `<h3>${title}</h3>${content}`;
        return section;
    }

    // Function to create data table
    function createDataTable(data) {
        if (!data || data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const rows = data.slice(0, 10).map(row => 
            `<tr>${Object.values(row).map(value => `<td>${value}</td>`).join('')}</tr>`
        ).join('');

        return `
            <div class="table-container">
                <table>
                    <thead>
                        <tr>${headers.map(header => `<th>${header}</th>`).join('')}</tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    // Function to display results
    function displayResults(result) {
        clearResults();

        // Display SQL Query
        if (result.sql_query) {
            const sqlSection = createSection('SQL Query', `<pre><code>${result.sql_query}</code></pre>`);
            resultsContainer.appendChild(sqlSection);
        }

        // Display Insights
        if (result.insights && result.insights.length > 0) {
            const insightsHtml = `<ul>${result.insights.map(insight => `<li>${insight}</li>`).join('')}</ul>`;
            const insightsSection = createSection('Insights', insightsHtml);
            resultsContainer.appendChild(insightsSection);
        }

        // Display Chart
        if (result.chart) {
            const chartSection = createSection('Chart', '<div id="chartContainer"></div>');
            resultsContainer.appendChild(chartSection);

            const layout = {
                ...result.chart.layout,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter, sans-serif' },
                margin: { t: 30, r: 20, b: 40, l: 60 },
                xaxis: {
                    ...result.chart.layout.xaxis,
                    gridcolor: '#f0f0f0',
                    linecolor: '#e5e7eb',
                    tickfont: { size: 12 }
                },
                yaxis: {
                    ...result.chart.layout.yaxis,
                    gridcolor: '#f0f0f0',
                    linecolor: '#e5e7eb',
                    tickfont: { size: 12 }
                }
            };

            Plotly.newPlot('chartContainer', result.chart.data, layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            });
        }

        // Display Data Preview
        if (result.data && result.data.length > 0) {
            const tableHtml = createDataTable(result.data);
            const dataSection = createSection('Data Preview', tableHtml);
            resultsContainer.appendChild(dataSection);
        }
    }

    // Handle form submission
    async function handleAnalyze(event) {
        event.preventDefault();
        
        const question = questionInput.value.trim();
        const databaseType = databaseSelect.value;

        if (!question) {
            showError('Please enter a question');
            return;
        }

        setLoading(true);
        clearResults();

        try {
            console.log('Sending request to backend...');
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    database_type: databaseType
                })
            });

            console.log('Received response from backend');
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Analysis failed');
            }

            displayResults(result);
        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message);
        } finally {
            setLoading(false);
        }
    }

    // Add event listeners
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', handleAnalyze);
    } else {
        console.error('Form element not found!');
    }
});
