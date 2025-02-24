<template>
  <div class="space-y-4">
    <div class="flex gap-4">
      <select 
        v-model="chartType" 
        class="p-2 border rounded-lg"
      >
        <option value="bar">Bar Chart</option>
        <option value="line">Line Chart</option>
      </select>
      
      <select 
        v-model="selectedX" 
        class="p-2 border rounded-lg"
      >
        <option value="">Select X-Axis</option>
        <option 
          v-for="(col, idx) in columns" 
          :key="idx" 
          :value="idx"
        >
          {{ col }}
        </option>
      </select>
      
      <select 
        v-model="selectedY" 
        class="p-2 border rounded-lg"
      >
        <option value="">Select Y-Axis</option>
        <option 
          v-for="(col, idx) in columns" 
          :key="idx" 
          :value="idx"
        >
          {{ col }}
        </option>
      </select>
    </div>

    <div 
      v-if="selectedX && selectedY"
      class="bg-white p-4 rounded-lg"
      style="height: 400px;"
    >
      <Bar
        v-if="chartType === 'bar'"
        :data="chartData"
        :options="chartOptions"
      />
      <Line
        v-if="chartType === 'line'"
        :data="chartData"
        :options="chartOptions"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { Bar, Line } from 'vue-chartjs'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

const props = defineProps({
  data: {
    type: Array,
    required: true
  },
  columns: {
    type: Array,
    required: true
  }
})

const chartType = ref('bar')
const selectedX = ref('')
const selectedY = ref('')

const chartData = computed(() => {
  if (!selectedX.value || !selectedY.value) return null

  const xIndex = parseInt(selectedX.value)
  const yIndex = parseInt(selectedY.value)

  return {
    labels: props.data.map(row => row[xIndex]?.toString() || ''),
    datasets: [{
      label: props.columns[yIndex],
      data: props.data.map(row => Number(row[yIndex]) || 0),
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1
    }]
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top'
    },
    title: {
      display: true,
      text: 'Data Visualization'
    }
  }
}
</script>
