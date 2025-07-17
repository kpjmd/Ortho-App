import { useEffect, useRef } from 'react';

export const useRealTimeUpdates = (callback, interval = 30000, enabled = true) => {
  const intervalRef = useRef(null);
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const startInterval = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        callbackRef.current();
      }, interval);
    };

    startInterval();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [interval, enabled]);

  const forceUpdate = () => {
    if (callbackRef.current) {
      callbackRef.current();
    }
  };

  return { forceUpdate };
};

export const usePeriodicRefresh = (refreshFunction, interval = 30000, dependencies = []) => {
  const intervalRef = useRef(null);

  useEffect(() => {
    const startInterval = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(refreshFunction, interval);
    };

    startInterval();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [...dependencies, interval]);

  return {
    clearInterval: () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    },
    restartInterval: () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(refreshFunction, interval);
    }
  };
};

export const useTabVisibility = (onVisible, onHidden) => {
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (onHidden) onHidden();
      } else {
        if (onVisible) onVisible();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [onVisible, onHidden]);
};