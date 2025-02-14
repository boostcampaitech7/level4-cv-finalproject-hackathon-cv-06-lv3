// components/Charts/HorizontalBarChart.jsx
import React from 'react';
import Chart from 'react-apexcharts';

function HorizontalBarChart({ data, colorsArray }) {
  // data: [{ label: 'Category A', value: 10 }, { label: 'Category B', value: 5 }, ...]
  const series = [
    {
      data: data.map(item => item.value),
    },
  ];

  const options = {
    chart: {
      type: 'bar',
      toolbar: { show: false },
    },
    plotOptions: {
      bar: {
        horizontal: true, // 가로 막대형
        distributed: true,
        barHeight: '70%',
      },
    },
    dataLabels: {
      enabled: true,
      style: {
        fontSize: '12px',
        colors: ['#fff'],
      },
      formatter: function (val) {
        return `${val}`;
      },
    },
    legend: {
      show: false,
    },
    xaxis: {
      categories: data.map(item => item.label),
      labels: {
        style: { colors: '#fff', fontSize: '10px' },
      },
    },
    yaxis: {
      labels: {
        style: { colors: '#fff' },
      },
    },
    colors: colorsArray || [
      'rgba(207, 158, 239, 1)',
      'rgba(199, 163, 239, 0.9)',
      'rgba(191, 169, 239, 0.8)',
      'rgba(184, 174, 239, 0.7)',
      'rgba(176, 180, 239, 0.6)',
      'rgba(168, 185, 240, 0.5)',
      'rgba(160, 191, 240, 0.6)',
      'rgba(153, 196, 240, 0.7)',
      'rgba(145, 202, 240, 0.8)',
      'rgba(137, 207, 240, 0.9)',
    ],
    tooltip: {
      theme: 'dark',
      x: { show: true },
    },
  };

  return <Chart options={options} series={series} type="bar" height="300px" />;
}

export default HorizontalBarChart;
