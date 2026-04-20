import { useEffect, useMemo } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Loader2 } from "lucide-react";

import { useAuthStore } from "@/stores/auth.store";
import { STORAGE_KEYS } from "@/config/api";
import AdminLayout from "@/layouts/admin-layout";

export function AdminRoute() {
  const location = useLocation();
  const { isLoading, isAuthenticated, checkAuth, user, setLoading } = useAuthStore();

  const hasToken = useMemo(() => {
    return !!user || !!localStorage.getItem(STORAGE_KEYS.accessToken);
  }, [user]);

  useEffect(() => {
    if (!hasToken && isLoading) { setLoading(false); return; }
    if (hasToken && !isAuthenticated) { checkAuth(); }
  }, [hasToken, isAuthenticated, isLoading, checkAuth, setLoading]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 animate-spin text-primary" />
          <p className="mt-4 text-default-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!hasToken || !isAuthenticated) {
    return <Navigate replace state={{ from: location }} to="/login" />;
  }

  if (user?.role !== "admin") {
    return <Navigate replace to="/login" />;
  }

  return <AdminLayout />;
}
