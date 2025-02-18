<template>
  <div class="container mx-auto p-4">
    <h1 class="text-3xl font-bold mb-6">AI Data Analyst</h1>
    
    <!-- Query Input -->
    <div class="mb-6">
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
          class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
      <div class="bg-gray-50 p-4 rounded-lg">
        <h3 class="font-semibold text-lg mb-2">Generated SQL Query</h3>
        <pre class="bg-gray-800 text-white p-4 rounded overflow-x-auto">{{ result.query }}</pre>
        
        <h3 class="font-semibold text-lg mt-4 mb-2">Query Explanation</h3>
        <p class="text-gray-700">{{ result.explanation }}</p>
      </div>

      <!-- Data Table -->
      <div v-if="result.rows && result.rows.length" class="bg-white rounded-lg shadow overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th 
                v-for="column in result.columns" 
                :key="column"
                class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {{ column }}
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="(row, index) in result.rows" :key="index">
              <td 
                v-for="(value, colIndex) in row" 
                :key="colIndex"
                class="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
              >
                {{ formatValue(value) }}
              </td>
            </tr>
          </tbody>
        </table>
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
</template>

<script setup>
import { ref } from 'vue'

const question = ref('')
const loading = ref(false)
const error = ref(null)
const result = ref(null)

const formatValue = (value) => {
  if (value === null) return 'NULL'
  if (typeof value === 'object') return JSON.stringify(value)
  return value.toString()
}

const analyzeData = async () => {
  if (!question.value.trim()) return
  
  loading.value = true
  error.value = null
  result.value = null
  
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
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Failed to analyze data')
    }
    
    result.value = await response.json()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>
