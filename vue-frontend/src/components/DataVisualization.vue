<template>
  <div class="space-y-4">
    <!-- Display Controls -->
    <div class="flex gap-4">
      <select 
        v-model="displayMode" 
        class="p-2 border rounded-lg"
      >
        <option value="table">Table View</option>
        <option value="bar">Bar Chart</option>
        <option value="line">Line Chart</option>
      </select>
    </div>

    <!-- Table View -->
    <div v-if="displayMode === 'table'" class="overflow-x-auto">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th 
              v-for="col in columns" 
              :key="col"
              class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            >
              {{ col }}
            </th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
          <tr v-for="(row, idx) in data" :key="idx">
            <td 
              v-for="col in columns" 
              :key="col"
              class="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
            >
              {{ formatValue(row[col]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Chart View -->
    <div 
      v-else
      class="bg-white rounded-lg"
    >
      <div class="flex flex-wrap gap-4 mb-4">
        <select 
          v-model="displayMode" 
          class="p-2 border rounded-lg"
        >
          <option value="bar">Bar Chart</option>
          <option value="line">Line Chart</option>
        </select>
        
        <select 
          v-model="chartConfig.xAxis" 
          class="p-2 border rounded-lg"
        >
          <option value="">Select X-Axis</option>
          <option v-for="col in columns" :key="col" :value="col">{{ col }}</option>
        </select>
        
        <select 
          v-model="chartConfig.yAxis" 
          class="p-2 border rounded-lg"
        >
          <option value="">Select Y-Axis</option>
          <option v-for="col in numericColumns" :key="col" :value="col">{{ col }}</option>
        </select>
      </div>

      <div 
        v-if="chartConfig.xAxis && chartConfig.yAxis"
        class="h-[400px] w-full"
      >
        <Bar
          v-if="displayMode === 'bar'"
          :data="chartData"
          :options="chartOptions"
        />
        <Line
          v-if="displayMode === 'line'"
          :data="chartData"
          :options="chartOptions"
        />
      </div>
      <div v-else class="text-center text-gray-500 py-4">
        Please select both X and Y axes to display the chart
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
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
} from 'chart.js';
import { Bar, Line } from 'vue-chartjs';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const props = defineProps({
  data: {
    type: Array,
    required: true
  }
});

const displayMode = ref('table');
const chartConfig = ref({
  xAxis: '',
  yAxis: ''
});

// Computed properties
const columns = computed(() => {
  if (!props.data || props.data.length === 0) return [];
  return Object.keys(props.data[0]);
});

const numericColumns = computed(() => {
  if (!props.data || props.data.length === 0) return [];
  return columns.value.filter(col => {
    const value = props.data[0][col];
    return typeof value === 'number' || 
           (typeof value === 'string' && !isNaN(parseFloat(value))) ||
           (typeof value === 'string' && value.match(/^[\d,.]+$/));
  });
});

const chartData = computed(() => {
  if (!props.data || !chartConfig.value.xAxis || !chartConfig.value.yAxis) {
    return null;
  }

  const xAxisKey = chartConfig.value.xAxis;
  const yAxisKey = chartConfig.value.yAxis;

  const labels = props.data.map(row => formatValue(row[xAxisKey]));
  const data = props.data.map(row => {
    const value = row[yAxisKey];
    if (typeof value === 'string') {
      // Handle string numbers with commas
      return parseFloat(value.replace(/,/g, ''));
    }
    return value;
  });

  const backgroundColor = [
    'rgba(54, 162, 235, 0.5)',
    'rgba(255, 99, 132, 0.5)',
    'rgba(75, 192, 192, 0.5)',
    'rgba(255, 206, 86, 0.5)',
    'rgba(153, 102, 255, 0.5)',
    'rgba(255, 159, 64, 0.5)'
  ];

  const borderColor = backgroundColor.map(color => color.replace('0.5', '1'));

  return {
    labels,
    datasets: [{
      label: yAxisKey,
      data,
      backgroundColor: displayMode.value === 'bar' ? backgroundColor : borderColor[0],
      borderColor: displayMode.value === 'bar' ? borderColor : borderColor[0],
      borderWidth: 1,
      tension: 0.1
    }]
  };
});

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
    },
    title: {
      display: true,
      text: 'Data Visualization'
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      ticks: {
        callback: (value) => new Intl.NumberFormat().format(value)
      }
    }
  }
};

// Helper function to format values
const formatValue = (value) => {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    return new Intl.NumberFormat().format(value);
  }
  if (typeof value === 'string' && value.includes('T') && !isNaN(Date.parse(value))) {
    return new Date(value).toLocaleDateString();
  }
  return String(value);
};
</script>
