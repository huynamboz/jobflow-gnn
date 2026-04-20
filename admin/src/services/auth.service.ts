import type { LoginRequest, LoginResponse, User, MeResponse, ChangePasswordRequest } from "@/types/auth.types";

import { apiClient } from "@/lib/api-client";

class AuthService {
  async login(data: LoginRequest): Promise<{ access: string; refresh: string; user: User }> {
    // Step 1: get tokens
    const tokenRes = await apiClient.post<LoginResponse>("/auth/login/", data);
    apiClient.setTokens(tokenRes.data.access, tokenRes.data.refresh);

    // Step 2: get user info
    const meRes = await apiClient.get<MeResponse>("/auth/me/");
    return {
      access: tokenRes.data.access,
      refresh: tokenRes.data.refresh,
      user: meRes.data.data,
    };
  }

  async getCurrentUser(): Promise<User> {
    const res = await apiClient.get<MeResponse>("/auth/me/");
    return res.data.data;
  }

  async changePassword(data: ChangePasswordRequest): Promise<void> {
    await apiClient.post("/auth/change-password/", data);
  }

  isAuthenticated(): boolean {
    return !!apiClient.getAccessToken();
  }
}

export const authService = new AuthService();
