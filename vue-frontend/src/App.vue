<template>
  <div class="min-h-screen bg-gray-100 py-6">
    <div class="container mx-auto px-4">
      <h1 class="text-3xl font-bold mb-6 text-gray-900">AI Data Analyst</h1>
      
      <!-- Query Input -->
      <div class="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div class="flex gap-2">
          <input 
            v-model="question" 
            @keyup.enter="analyzeData"
            type="text" 
            placeholder="Ask a question about your data..." 
            class="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
          <button 
            @click="analyzeData" 
            class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            :disabled="loading"
          >
            {{ loading ? 'Analyzing...' : 'Analyze' }}
          </button>
        </div>
        
        <!-- Loading State -->
        <div v-if="loading" class="mt-4">
          <div class="animate-pulse flex space-x-4">
            <div class="flex-1 space-y-4 py-1">
              <div class="h-4 bg-gray-200 rounded w-3/4"></div>
              <div class="space-y-2">
                <div class="h-4 bg-gray-200 rounded"></div>
                <div class="h-4 bg-gray-200 rounded w-5/6"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Error Message -->
      <div v-if="error" class="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
        {{ error }}
      </div>

      <!-- Results -->
      <div v-if="result" class="space-y-6">
        <!-- Query Info -->
        <div class="bg-white rounded-lg shadow-sm p-6">
          <h3 class="font-semibold text-lg mb-2">Generated SQL Query</h3>
          <pre class="bg-gray-800 text-white p-4 rounded overflow-x-auto text-sm">{{ result.query }}</pre>
          
          <template v-if="result.performance">
            <h3 class="font-semibold text-lg mt-6 mb-2">Query Performance</h3>
            <div class="text-gray-700">
              <p>Execution Time: {{ (result.performance.execution_time_ms / 1000).toFixed(2) }}s</p>
              <p>Query Complexity: {{ result.performance.query_complexity }}</p>
              <p>Metrics Used: {{ result.performance.metrics_used.join(', ') }}</p>
            </div>
          </template>
        </div>

        <!-- Data Display -->
        <div v-if="result.results && result.results.length > 0" class="bg-white rounded-lg shadow-sm p-6">
          <DataVisualization 
            :data="result.results"
          />
        </div>
        
        <!-- No Results Message -->
        <div 
          v-else-if="result.message" 
          class="text-center p-4 bg-gray-50 rounded-lg text-gray-700"
        >
          {{ result.message }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import DataVisualization from './components/DataVisualization.vue'

// State
const question = ref('')
const loading = ref(false)
const error = ref(null)
const result = ref(null)

// Helper function to format values
const formatValue = (value) => {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'number') {
    return new Intl.NumberFormat().format(value)
  }
  return String(value)
}

// Function to analyze data
async function analyzeData() {
  if (!question.value.trim()) return
  
  loading.value = true
  error.value = null
  
  try {
    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question.value
      })
    })
    
    const data = await response.json()
    
    if (!response.ok) {
      throw new Error(data.detail?.error || data.detail || 'Failed to analyze data')
    }
    
    result.value = data
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>
