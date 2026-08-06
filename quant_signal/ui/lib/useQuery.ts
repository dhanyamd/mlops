"use client";

import { useEffect, useState } from "react";

/**
 * Load data from the API for a given dependency key.
 *
 * `loading` is derived (the loaded key differs from the requested key), so no
 * state is set synchronously inside the effect — the new React lint rules are
 * satisfied, and a stale result stays visible while the next one loads.
 */
export function useQuery<T>(deps: unknown[], loader: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const requestKey = JSON.stringify(deps);

  useEffect(() => {
    let cancelled = false;
    loader()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError(null);
          setLoadedKey(requestKey);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setData(null);
          setError(e instanceof Error ? e.message : String(e));
          setLoadedKey(requestKey);
        }
      });
    return () => {
      cancelled = true;
    };
    // loader is a fresh closure per render; only its deps define the key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey]);

  return { data, error, loading: loadedKey !== requestKey };
}
