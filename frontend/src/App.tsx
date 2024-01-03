import { useEffect, useState } from "react";
import { useDebounce } from "@uidotdev/usehooks";
import ReactTextareaAutosize from "react-textarea-autosize";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
} from "@tanstack/react-query";

import { MdFeedback } from "react-icons/md";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

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
  const [feedbackMode, setFeedbackMode] = useState<boolean>(false);
  const [feedbackModeInput, setFeedbackModeInput] = useState<string>("");
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
  const updateFeedbackMutation = useMutation({
    mutationFn: async (feedback: { id: string; feedback: string }) => {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(feedback),
      });
      const resJson = await res.json();
      if (!res.ok) throw new Error(resJson.error);
      return resJson;
    },
    onSuccess: () => {
      toast.success("Successfully submitted feedback!");
      setFeedbackMode(false);
    },
  });
  useEffect(() => {
    if (updateFeedbackMutation.error) {
      toast.error(updateFeedbackMutation.error.message);
    }
  }, [updateFeedbackMutation.error]);

  useEffect(() => {
    if (debouncedUserInput) {
      refetch();
    }
  }, [refetch, debouncedUserInput]);
  const isLoading = debouncedUserInput && isFetching;
  const showFeedbackMode = data && data.output && feedbackMode;
  if (!debouncedUserInput && data && data.output) {
    data.output = "";
  }

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
              disabled={feedbackMode}
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
                value={
                  data && data.output
                    ? feedbackMode
                      ? feedbackModeInput
                      : data.output
                    : ""
                }
                onChange={(e) => {
                  if (showFeedbackMode) {
                    setFeedbackModeInput(e.target.value);
                  }
                }}
                minRows={10}
                maxRows={30}
                disabled={!feedbackMode}
                autoComplete="off"
                className={
                  "resize-none border border-slate-400 bg-slate-200 rounded-lg p-2 w-full" +
                  (isLoading ? " animate-loading-pulsate" : "")
                }
              />
              {showFeedbackMode ? (
                <div className="absolute bottom-4 right-2 ">
                  <button
                    className="mr-4 bg-slate-300 p-2 rounded-md"
                    onClick={() => setFeedbackMode(false)}
                  >
                    Cancel
                  </button>
                  <button
                    className="text-blue-600 bg-slate-300 p-2 rounded-md"
                    onClick={() => {
                      updateFeedbackMutation.mutate({
                        id: data.id,
                        feedback: feedbackModeInput,
                      });
                    }}
                  >
                    Submit
                  </button>
                </div>
              ) : (
                <div className="absolute bottom-2 right-2">
                  <button
                    onClick={() => {
                      setFeedbackMode(true);
                      setFeedbackModeInput(data.output);
                    }}
                  >
                    <MdFeedback size="20" />
                  </button>
                </div>
              )}
            </div>
            {showFeedbackMode && (
              <p className="text-sm bg-slate-300 rounded-lg p-2">
                Submit an improved translation so that all users can benefit
                from a better experience. The feedback will be reviewed and make
                the translator better.
              </p>
            )}
          </div>
        </div>
      </div>
      <div className="m-4 mt-8 p-4 bg-white rounded-lg">
        <p className="text-md italic">
          Last updated:{" "}
          {data && data.last_updated
            ? new Date(data.last_updated * 1000).toLocaleString()
            : "..."}
        </p>
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
      <ToastContainer />
    </div>
  );
}

export default AppWrapper;
