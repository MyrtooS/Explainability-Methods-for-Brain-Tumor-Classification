import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Generate a linear interpolation between the baseline and the original image.
def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x

  images = baseline_x +  alphas_x * delta

  return images

# Calculate gradients in order to measure the relationship between
# changes to a feature and changes in the model's predictions
def compute_gradients(images, target_class_idx,model):
    with tf.GradientTape() as tape:
      tape.watch(images)
      logits = model(images)
      probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)

  
#  Computing the numerical approximation of an integral for Integrated Gradients
def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients


# Combine the 3 previous general parts together into an IntegratedGradients function 
def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         model,
                         m_steps=50,
                         batch_size=32,
                         ):
  # Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Collect gradients.    
  gradient_batches = []
    
  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx,model = model)
    gradient_batches.append(gradient_batch)
      
  # Stack path gradients together row-wise into single tensor.
  total_gradients = tf.stack(gradient_batch)

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx,model):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx,model=model)
    return gradient_batch


# Visualize attributions, and overlay them on the original image
def plot(baseline,
                          image,
                          target_class_idx,
                          model,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):

  attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      model = model,
                                      m_steps=m_steps)
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

  fig = plt.figure(figsize=(5, 5))
  plt.imshow(attribution_mask, cmap=cmap)

  return attributions, attribution_mask

 
