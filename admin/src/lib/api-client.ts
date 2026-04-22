import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  InternalAxiosRequestConfig,
} from "axios";

import { API_CONFIG, STORAGE_KEYS } from "@/config/api";

interface ApiErrorResponse {
  success: false;
  error: { code: string; message: string; status: number };
}

// Queue for requests during token refresh
let isRefreshing = false;
const failedQueue: Array<{
  resolve: (value?: unknown) => void;
  reject: (error?: unknown) => void;
}> = [];

const client: AxiosInstance = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: { "Content-Type": "application/json" },
});

function getAccessToken(): string | null {
  return localStorage.getItem(STORAGE_KEYS.accessToken);
}

function getRefreshToken(): string | null {
  return localStorage.getItem(STORAGE_KEYS.refreshToken);
}

function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(STORAGE_KEYS.accessToken, accessToken);
  localStorage.setItem(STORAGE_KEYS.refreshToken, refreshToken);
}

function clearTokens(): void {
  localStorage.removeItem(STORAGE_KEYS.accessToken);
  localStorage.removeItem(STORAGE_KEYS.refreshToken);
}

function extractErrorMessage(error: AxiosError<ApiErrorResponse>): Error {
  if (error.response) {
    const msg = error.response.data?.error?.message;
    return new Error(msg || "An error occurred");
  }
  if (error.request) return new Error("Network error. Please check your connection.");
  return new Error(error.message || "An unexpected error occurred");
}

function processQueue(error: unknown): void {
  failedQueue.forEach((p) => (error ? p.reject(error) : p.resolve()));
  failedQueue.length = 0;
}

// Request interceptor: attach token + handle FormData
client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = getAccessToken();
    if (token && config.headers) config.headers.Authorization = `Bearer ${token}`;
    // Let browser set Content-Type with boundary for multipart
    if (config.data instanceof FormData) {
      delete config.headers["Content-Type"];
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor: handle 401 + refresh
client.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ApiErrorResponse>) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(() => client.request(originalRequest));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = getRefreshToken();
        if (!refreshToken) throw new Error("No refresh token");

        const res = await axios.post<{ access: string }>(
          `${API_CONFIG.baseURL}/auth/refresh/`,
          { refresh: refreshToken },
        );

        const newAccess = res.data.access;
        setTokens(newAccess, refreshToken);
        processQueue(null);

        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${newAccess}`;
        }
        return client.request(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError);
        clearTokens();
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(extractErrorMessage(error));
  },
);

export const apiClient = {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
  get:    <T>(url: string, config?: AxiosRequestConfig) => client.get<T>(url, config),
  post:   <T>(url: string, data?: unknown, config?: AxiosRequestConfig) => client.post<T>(url, data, config),
  put:    <T>(url: string, data?: unknown, config?: AxiosRequestConfig) => client.put<T>(url, data, config),
  patch:  <T>(url: string, data?: unknown, config?: AxiosRequestConfig) => client.patch<T>(url, data, config),
  delete: <T>(url: string, config?: AxiosRequestConfig) => client.delete<T>(url, config),
};
