import { useMemo } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuthStore } from "@/stores/auth.store";
import { STORAGE_KEYS } from "@/config/api";

export function PublicRoute() {
  const location = useLocation();
  const { isAuthenticated, user } = useAuthStore();

  const hasToken = useMemo(() => {
    return !!user || !!localStorage.getItem(STORAGE_KEYS.accessToken);
  }, [user]);

  if (hasToken && isAuthenticated) {
    const from = (location.state as { from?: Location })?.from?.pathname || "/admin";
    return <Navigate replace to={from} />;
  }

  return <Outlet />;
}
