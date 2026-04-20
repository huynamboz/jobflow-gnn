import { create } from "zustand";

import type { LabelingCV, LabelingProgress, PairQueueItem, QueueResponse } from "@/types/labeling.types";

interface LabelingState {
  currentCV: LabelingCV | null;
  pairs: PairQueueItem[];
  progress: LabelingProgress | null;
  activePairId: number | null;
  isLoading: boolean;
  isEmpty: boolean; // all pairs labeled
}

interface LabelingActions {
  setQueue: (data: QueueResponse) => void;
  setEmpty: () => void;
  setLoading: (v: boolean) => void;
  setActivePair: (id: number | null) => void;
  markLabeled: (pairId: number) => void;
  markSkipped: (pairId: number) => void;
  reset: () => void;
}

type LabelingStore = LabelingState & LabelingActions;

export const useLabelingStore = create<LabelingStore>((set, get) => ({
  currentCV: null,
  pairs: [],
  progress: null,
  activePairId: null,
  isLoading: false,
  isEmpty: false,

  setQueue: (data) => {
    const firstPendingId = data.pairs[0]?.id ?? null;
    set({
      currentCV: data.cv,
      pairs: data.pairs,
      progress: data.progress,
      activePairId: firstPendingId,
      isEmpty: false,
      isLoading: false,
    });
  },

  setEmpty: () => set({ isEmpty: true, isLoading: false, currentCV: null, pairs: [] }),

  setLoading: (isLoading) => set({ isLoading }),

  setActivePair: (id) => set({ activePairId: id }),

  markLabeled: (pairId) => {
    const { pairs, progress } = get();
    const remaining = pairs.filter((p) => p.id !== pairId);
    const nextId = remaining[0]?.id ?? null;
    set({
      pairs: remaining,
      activePairId: nextId,
      progress: progress
        ? { ...progress, labeled: progress.labeled + 1, pending: progress.pending - 1, current_cv_pending: progress.current_cv_pending - 1 }
        : null,
    });
  },

  markSkipped: (pairId) => {
    const { pairs, progress } = get();
    const remaining = pairs.filter((p) => p.id !== pairId);
    const nextId = remaining[0]?.id ?? null;
    set({
      pairs: remaining,
      activePairId: nextId,
      progress: progress
        ? { ...progress, skipped: progress.skipped + 1, pending: progress.pending - 1, current_cv_pending: progress.current_cv_pending - 1 }
        : null,
    });
  },

  reset: () => set({ currentCV: null, pairs: [], progress: null, activePairId: null, isLoading: false, isEmpty: false }),
}));
