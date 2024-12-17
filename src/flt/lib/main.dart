import 'package:flutter/material.dart';

void main() {
  runApp(
    const Center(
      child: Text(
        'Hello',
        textDirection: TextDirection.ltr,
        style: TextStyle(
          color: Colors.red,
        ),
      ),
    ),
  );
}
