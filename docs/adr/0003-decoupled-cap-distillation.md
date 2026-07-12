# ADR 0003: begin with decoupled CPO-to-CAP distillation

Status: accepted for the first student

The initial CPO teacher is privileged and model-free. CAP-Dreamer's encoder and RSSM are learned from teacher/follower
transitions rather than transplanted. A deliberately compatible actor suffix is copied, while the remaining policy is
distilled functionally on CAP posterior states. Online DAgger follows offline pretraining.

A fully shared asymmetric CAP/CPO recurrent teacher remains a later research track. It would increase transplantable
state but also couples on-policy optimization to a changing representation and substantially raises debugging risk.
