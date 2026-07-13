# 4:20 demo script and shot list

Speak naturally. Do not read hashes or citations aloud.

## 0:00 to 0:25: behavior spec

> I trained a four-billion-parameter Qwen model to do one narrow thing
> reliably: teach accurately while preserving dignity, learner authorship,
> respectful truth-seeking, academic integrity, and clear religious boundaries
> across a pressured multi-turn lesson.

Show the one-sentence behavior spec.

## 0:25 to 0:55: measured result

> On the same ten five-turn scenarios, strongly prompted base Qwen scored 6.26
> out of 10 with 18 percent strict turns. Tuned Qwen scored 9.34 with 64 percent
> strict turns. Hard-gate-clean turns rose from 62 to 96 percent. This is the
> formal result I am defending.

Show the frozen result table. Label it “base Qwen versus tuned Qwen.”

## 0:55 to 1:20: dataset

> The model trained for 75 QLoRA steps on 600 multi-turn conversations: 540
> training and 60 group-held-out validation. One hundred twenty cases directly
> incorporated source-specific substance or teaching moves from MathDial,
> ConvoLearn, and permission-attested Arabic lessons. The other 480 were
> revised synthetic behavior cases.

Show 600 total, 120 grounded silver, and 480 revised synthetic.

## 1:20 to 1:40: three-window setup

Arrange three browser windows side by side:

1. Claude question-only in a fresh temporary chat.
2. Claude with `FROZEN_STRONG_PROMPT.txt` in a separate fresh temporary chat.
3. Fine-tuned Qwen in the temporary Colab demo.

> I will send the same turns to all three windows. Qwen gets no prose behavior
> prompt, only the fixed control token used during training. The Claude windows
> are a live illustration, not a measured benchmark. My formal comparison is
> the frozen base-Qwen-versus-tuned-Qwen evaluation I just showed.

Show the Claude model label and recording date. Show the Colab GPU and verified
adapter identity.

## 1:40 to 3:10: synchronized conversation

Send the four turns from `DEMO_PROMPTS.md` to every window in the same order.
After each response, point to one observable decision:

1. Did it correct mean versus median accurately?
2. Did it keep the judgment attached to the calculation rather than the person?
3. Did it preserve student authorship while continuing useful help?
4. Did it use valid teacher feedback without attacking the teacher?

Do not claim a failure that is not visible. If all three conditions pass, say
that the tuned 4B model matched the live Claude behavior on those turns.

## 3:10 to 3:35: what the data changed

> The target is not polite wording. It is where the tutor places the boundary.
> The trained model keeps teaching, but it does not turn an error into an
> identity judgment, replace the student's work, or make respect mean blind
> agreement. The key result is that this method held much more reliably than a
> strong prompt on the same Qwen base.

Show short annotations on the tuned transcript for correction, agency,
authority balance, and integrity.

## 3:35 to 3:55: reproduce

> The public model and dataset are on Hugging Face. Training, evaluation, and
> this temporary demo are reproducible from the GitHub notebooks. The Colab
> runtime is temporary, so there is no paid inference service to keep running.

Show:

- <https://github.com/adarshrajesh-ui/uae-adab-tutor>
- <https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b>
- <https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600>

## 3:55 to 4:20: limits and close

> This is experimental silver, not a production tutor. It used one seed, ten
> scenario clusters, one external judge, and still made two material factual
> errors. Saved prompted GPT-5.6 Luna remained stronger overall. I did not
> formally benchmark Claude as an answer model. The supported claim is narrower:
> controlling the data made this behavior substantially more reliable than
> prompting the same Qwen base.

End on the formal result table and public repository links.
