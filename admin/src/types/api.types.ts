export interface ApiSuccess<T> {
  success: true;
  data: T;
  count?: number;
}

export interface ApiError {
  success: false;
  error: {
    code: string;
    message: string;
    status: number;
  };
}

export type ApiResponse<T> = ApiSuccess<T> | ApiError;
