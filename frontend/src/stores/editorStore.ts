import { create } from 'zustand'

type EditorState = {
  maskBlob: Blob | null
  referenceFiles: File[]
  setMaskBlob: (blob: Blob | null) => void
  addReferenceFiles: (files: FileList | File[]) => void
  clearReferences: () => void
  reset: () => void
}

export const useEditorStore = create<EditorState>((set) => ({
  maskBlob: null,
  referenceFiles: [],
  setMaskBlob: (maskBlob) => set({ maskBlob }),
  addReferenceFiles: (files) =>
    set((state) => {
      const next = [...state.referenceFiles]
      for (const file of Array.from(files)) {
        if (next.length >= 10) break
        next.push(file)
      }
      return { referenceFiles: next }
    }),
  clearReferences: () => set({ referenceFiles: [] }),
  reset: () => set({ maskBlob: null, referenceFiles: [] }),
}))
