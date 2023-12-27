import { useEffect, useState } from "react";
import { useDebounce } from "@uidotdev/usehooks";
import ReactTextareaAutosize from "react-textarea-autosize";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";

const queryClient = new QueryClient();

function AppWrapper() {
  return (
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

function App() {
  const [userInput, setUserInput] = useState<string>("");
  const debouncedUserInput = useDebounce<string>(userInput, 1000);
  const { isFetching, error, data, refetch } = useQuery({
    enabled: false,
    retry: false,
    queryKey: ["translation"],
    queryFn: async () => {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: userInput,
        }),
      });
      const resJson = await res.json();
      if (!res.ok) throw new Error(resJson.error);
      return resJson;
    },
  });

  useEffect(() => {
    if (debouncedUserInput) {
      refetch();
    }
  }, [refetch, debouncedUserInput]);
  const isLoading = debouncedUserInput && isFetching;

  return (
    <div className="bg-slate-100 h-screen">
      <div className="p-4 mb-4 bg-white border-b-2">
        English to French Neural Translation
      </div>
      <div className="pr-4 pl-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="mb-2">English</p>
            <ReactTextareaAutosize
              value={userInput}
              minRows={10}
              maxRows={30}
              onChange={(e) => setUserInput(e.target.value)}
              autoComplete="off"
              autoFocus={true}
              className={
                "resize-none border bg-white rounded-lg p-2 w-full" +
                (error ? " border-red-500" : " border-slate-400")
              }
            />
            <p
              className={
                "border rounded-lg p-2 border-red-500 bg-red-200" +
                (error ? "" : " hidden")
              }
            >
              {error ? error.message : "This is a placeholder error"}
            </p>
          </div>
          <div>
            <p className="mb-2">French</p>
            <div className="relative">
              <ReactTextareaAutosize
                value={data && data.output ? data.output : ""}
                minRows={10}
                maxRows={30}
                disabled
                autoComplete="off"
                className={
                  "resize-none border border-slate-400 bg-slate-200 rounded-lg p-2 w-full" +
                  (isLoading ? " animate-loading-pulsate" : "")
                }
              />
            </div>
          </div>
        </div>
      </div>
      <div className="m-4 mt-8 p-4 bg-white rounded-lg">
        <p className="mb-1 text-lg">Features</p>
        <p className="text-sm">This English to French translation tool uses a Transformer model implemented in Pytorch</p>
      </div>
    </div>
  );
}

export default AppWrapper;
