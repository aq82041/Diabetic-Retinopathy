from sklearn.metrics import classification_report, confusion_matrix

model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)
					
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['exudates', 'non_exudates']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))