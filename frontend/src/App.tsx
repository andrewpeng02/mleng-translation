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
  const debouncedUserInput = useDebounce<string>(userInput, 250);
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
        English to French Translation
      </div>
      <div className="pr-4 pl-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
        <p className="text-md">
          This is an English to French translation tool uses a Transformer model
          implemented in Pytorch.{" "}
        </p>
        <p className="mb-1 mt-4 text-lg">Features</p>
        <ul className="ml-4 list-disc">
          <li>
            <p className="text-sm">
              It automatically retrains the model weekly with a pipeline created
              with Prefect using new training data and user feedback.{" "}
            </p>
          </li>
          <li>
            <p className="text-sm">
              The model and outputs are continuously monitored using Prometheus
              and Grafana to prevent model degradation and ensure quality
              responses.{" "}
            </p>
          </li>
          <li>
            <p className="text-sm">
              Model compression techniques such as quantization allows inference
              to be done on CPU in under 200ms
            </p>
          </li>
          <li>
            <p className="text-sm">
              This project was created by{" "}
              <a
                href="https://www.linkedin.com/in/peng-andrew/"
                className="text-blue-600 hover:underline hover:text-blue-800"
              >
                Andrew Peng
              </a>
              , and the source code is available{" "}
              <a
                href="https://github.com/andrewpeng02/mleng-translation"
                className="text-blue-600 hover:underline hover:text-blue-800"
              >
                here
              </a>{" "}
            </p>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default AppWrapper;
