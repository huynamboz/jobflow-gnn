import type { User } from "@/types/auth.types";

import { create } from "zustand";

import { apiClient } from "@/lib/api-client";
import { authService } from "@/services/auth.service";
import { STORAGE_KEYS } from "@/config/api";

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

interface AuthActions {
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
}

type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  isLoading: true,
  isAuthenticated: false,

  setUser: (user) => set({ user, isAuthenticated: !!user }),
  setLoading: (isLoading) => set({ isLoading }),

  checkAuth: async () => {
    const token = localStorage.getItem(STORAGE_KEYS.accessToken);
    if (!token) {
      set({ user: null, isLoading: false, isAuthenticated: false });
      return;
    }
    try {
      const user = await authService.getCurrentUser();
      set({ user, isLoading: false, isAuthenticated: true });
    } catch {
      apiClient.clearTokens();
      set({ user: null, isLoading: false, isAuthenticated: false });
    }
  },

  login: async (username, password) => {
    const { access, refresh, user } = await authService.login({ username, password });
    apiClient.setTokens(access, refresh);
    set({ user, isAuthenticated: true, isLoading: false });
  },

  logout: () => {
    apiClient.clearTokens();
    set({ user: null, isAuthenticated: false, isLoading: false });
    window.location.href = "/login";
  },
}));
