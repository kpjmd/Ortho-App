import { useMemo } from 'react';

export const useChartConfig = (type = 'line') => {
  const config = useMemo(() => {
    const baseConfig = {
      responsive: true,
      maintainAspectRatio: false,
      margin: { top: 20, right: 30, left: 20, bottom: 5 },
    };

    const colors = {
      primary: '#3b82f6',
      secondary: '#10b981',
      warning: '#f59e0b',
      danger: '#ef4444',
      info: '#06b6d4',
      purple: '#8b5cf6',
      pink: '#ec4899',
      gray: '#6b7280',
    };

    const chartSpecificConfig = {
      line: {
        stroke: colors.primary,
        strokeWidth: 2,
        dot: { r: 4 },
        activeDot: { r: 6 },
      },
      area: {
        fill: colors.primary,
        fillOpacity: 0.1,
        stroke: colors.primary,
        strokeWidth: 2,
      },
      bar: {
        fill: colors.primary,
      },
      pie: {
        colors: [colors.primary, colors.secondary, colors.warning, colors.danger, colors.info],
      },
      scatter: {
        fill: colors.primary,
      },
    };

    return {
      ...baseConfig,
      ...chartSpecificConfig[type],
      colors,
    };
  }, [type]);

  return config;
};

export const useScoreColors = () => {
  return useMemo(() => ({
    excellent: '#10b981',
    good: '#3b82f6',
    fair: '#f59e0b',
    poor: '#ef4444',
    getColorForScore: (score, maxScore = 100) => {
      const percentage = (score / maxScore) * 100;
      if (percentage >= 80) return '#10b981';
      if (percentage >= 60) return '#3b82f6';
      if (percentage >= 40) return '#f59e0b';
      return '#ef4444';
    },
  }), []);
};

export const useAlertColors = () => {
  return useMemo(() => ({
    critical: '#ef4444',
    high: '#f59e0b',
    medium: '#3b82f6',
    low: '#10b981',
    info: '#06b6d4',
    getColorForSeverity: (severity) => {
      const colors = {
        critical: '#ef4444',
        high: '#f59e0b',
        medium: '#3b82f6',
        low: '#10b981',
        info: '#06b6d4',
      };
      return colors[severity] || colors.info;
    },
  }), []);
};

export const useTimeSeriesConfig = (dataKeys = []) => {
  const chartConfig = useChartConfig('line');
  
  return useMemo(() => {
    const lineColors = [
      chartConfig.colors.primary,
      chartConfig.colors.secondary,
      chartConfig.colors.warning,
      chartConfig.colors.danger,
      chartConfig.colors.info,
      chartConfig.colors.purple,
      chartConfig.colors.pink,
    ];

    const lines = dataKeys.map((key, index) => ({
      key,
      stroke: lineColors[index % lineColors.length],
      strokeWidth: 2,
      dot: { r: 4 },
      activeDot: { r: 6 },
    }));

    return {
      ...chartConfig,
      lines,
      xAxis: {
        dataKey: 'date',
        tick: { fontSize: 12 },
        axisLine: { stroke: chartConfig.colors.gray },
      },
      yAxis: {
        tick: { fontSize: 12 },
        axisLine: { stroke: chartConfig.colors.gray },
      },
      grid: {
        strokeDasharray: '3 3',
        stroke: chartConfig.colors.gray,
        opacity: 0.3,
      },
    };
  }, [dataKeys, chartConfig]);
};

export const useBarChartConfig = (dataKeys = []) => {
  const chartConfig = useChartConfig('bar');
  
  return useMemo(() => {
    const barColors = [
      chartConfig.colors.primary,
      chartConfig.colors.secondary,
      chartConfig.colors.warning,
      chartConfig.colors.danger,
      chartConfig.colors.info,
    ];

    const bars = dataKeys.map((key, index) => ({
      key,
      fill: barColors[index % barColors.length],
    }));

    return {
      ...chartConfig,
      bars,
      xAxis: {
        dataKey: 'name',
        tick: { fontSize: 12 },
        axisLine: { stroke: chartConfig.colors.gray },
      },
      yAxis: {
        tick: { fontSize: 12 },
        axisLine: { stroke: chartConfig.colors.gray },
      },
      grid: {
        strokeDasharray: '3 3',
        stroke: chartConfig.colors.gray,
        opacity: 0.3,
      },
    };
  }, [dataKeys, chartConfig]);
};

export const usePieChartConfig = () => {
  const chartConfig = useChartConfig('pie');
  
  return useMemo(() => ({
    ...chartConfig,
    colors: chartConfig.colors.pie,
    label: {
      fontSize: 12,
      fill: '#374151',
    },
    legend: {
      verticalAlign: 'bottom',
      height: 36,
    },
  }), [chartConfig]);
};

export const useResponsiveChart = () => {
  return useMemo(() => ({
    mobile: {
      width: '100%',
      height: 250,
      margin: { top: 10, right: 10, left: 10, bottom: 10 },
    },
    tablet: {
      width: '100%',
      height: 300,
      margin: { top: 15, right: 20, left: 15, bottom: 15 },
    },
    desktop: {
      width: '100%',
      height: 400,
      margin: { top: 20, right: 30, left: 20, bottom: 20 },
    },
  }), []);
};