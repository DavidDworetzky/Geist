---
title: Using Pitchblend
description: Understand chat, models, privacy, and the relationship between Pitchblend and Geist.
order: 3
category: Product Guide
icon: book-open
---

# Using Pitchblend

Pitchblend is the desktop product experience. Geist is the engine underneath it. Geist manages model execution, conversations, tools, and local data; Pitchblend packages those capabilities for everyday use.

## Chat

Choose an available model, create a conversation, and send a message. Local models process prompts on your machine. Online providers require a network connection and send requests to the provider you configure.

## Models

The model paths available to you depend on your operating system and hardware:

- Apple silicon can use the MLX runner.
- Supported Windows and Linux systems can use managed llama.cpp with GGUF models.
- Compatible online providers can be configured separately.

## Privacy

Local inference keeps model processing on your computer. Configuring an online provider changes that boundary: prompts sent to that provider are subject to its network service and privacy terms.

## Pre-release limitations

Pitchblend is under active development. Public installers, automatic updates, and some product integrations described in planning material are not yet generally available.
