import 'package:flutter/material.dart';
import 'dart:math' as math;

class StatelessContainer extends StatelessWidget {
  final double width;
  final double height;
  final Color color;

  StatelessContainer({super.key, required this.height, required this.width})
      : color = Color((math.Random().nextDouble() * 0xFFFFFF).toInt())
            .withValues(alpha: 1.0);

  @override
  Widget build(BuildContext context) {
    return Container(width: width, height: height, color: color);
  }
}

class StatefulContainer extends StatefulWidget {
  final double width;
  final double height;

  const StatefulContainer(
      {super.key, required this.height, required this.width});

  @override
  State<StatefulContainer> createState() => _StatefulContainerState();
}

class _StatefulContainerState extends State<StatefulContainer> {
  final Color color;

  _StatefulContainerState()
      : color = Color((math.Random().nextDouble() * 0xFFFFFF).toInt())
            .withValues(alpha: 1.0);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: widget.width,
      height: widget.height,
      color: color,
    );
  }
}

class Screen extends StatefulWidget {
  const Screen({super.key});

  @override
  State<Screen> createState() {
    return _ScreenState();
  }
}

class _ScreenState extends State<Screen> {
  List<Widget> widgets = [
    StatefulContainer(key: UniqueKey(), width: 100, height: 100),
    StatefulContainer(key: UniqueKey(), width: 150, height: 150),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.center,
          children: widgets,
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: onPressed,
        child: Icon(Icons.undo),
      ),
    );
  }

  onPressed() {
    setState(() {
      widgets.insert(0, widgets.removeAt(1));
    });
  }
}

void main() {
  runApp(
    MaterialApp(
      home: Screen(),
    ),
  );
}
